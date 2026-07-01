"""Position-specific residual model backtest.

Default challenger model:
    position-specific raw APEX with the `profile` feature set

Production features remain experimental and are evaluated in ablation reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backtest import DEFAULT_APEX_PLUS_FACTOR, aggregate_report, flatten_metrics, score_apex_plus
from build_features import merge_optional_features
from feature_registry import ENRICHED_FEATURE_FILE, POSITION_MODEL_GROUPS, production_features_for_pos
from feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS, model_features_for
from pipeline import (
    ROOT,
    load_dataset,
    make_baseline,
    make_pick_baseline,
    make_resid,
    metric_row,
    prepare_fold,
    score_apex,
    tune_position_shrinkage,
)


def load_modeling_table(data_dir: str | None = None, enriched_path: str | None = None, end_year: int | None = None) -> pd.DataFrame:
    path = Path(enriched_path) if enriched_path else ENRICHED_FEATURE_FILE
    if enriched_path and path.exists():
        df = pd.read_csv(path)
        if end_year is not None and "Year" in df.columns:
            df = df[pd.to_numeric(df["Year"], errors="coerce") <= end_year].copy()
        return df
    base = load_dataset(data_dir=data_dir, end_year=end_year if end_year is not None else 2016)
    enriched, _ = merge_optional_features(base)
    return enriched


def position_family(pos_g: str) -> str:
    pos = str(pos_g)
    for family, positions in POSITION_MODEL_GROUPS.items():
        if pos in positions:
            return family
    return "OTHER"


def available_extra_features(train: pd.DataFrame, positions: tuple[str, ...], min_coverage: float) -> list[str]:
    wanted: list[str] = []
    for pos in positions:
        wanted.extend(production_features_for_pos(pos))
    out: list[str] = []
    for col in sorted(set(wanted)):
        if col not in train.columns:
            continue
        coverage = pd.to_numeric(train[col], errors="coerce").notna().mean()
        if coverage >= min_coverage:
            out.append(col)
    return out


def fit_position_residuals(
    train: pd.DataFrame,
    base: Callable[[pd.DataFrame], np.ndarray],
    *,
    feats: list[str],
    min_train_rows: int,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    global_resid, _ = make_resid(train, base, feats=feats)
    models = {}
    report = {"families": {}, "fallback": "global_profile_residual", "features": feats}

    for family, positions in POSITION_MODEL_GROUPS.items():
        mask = train["pos_g"].astype(str).isin(positions)
        subset = train.loc[mask].copy()
        if len(subset) < min_train_rows:
            report["families"][family] = {
                "trained": False,
                "n_train": int(len(subset)),
                "features": feats,
                "reason": f"n_train < {min_train_rows}",
            }
            continue
        resid, _ = make_resid(subset, base, feats=feats)
        models[family] = (positions, resid, feats)
        report["families"][family] = {
            "trained": True,
            "n_train": int(len(subset)),
            "features": feats,
        }

    def predict(part: pd.DataFrame) -> np.ndarray:
        out = global_resid(part)
        fam = part["pos_g"].astype(str).map(position_family)
        for family, (_, resid, _) in models.items():
            mask = fam.eq(family)
            if mask.any():
                out[mask.to_numpy()] = resid(part.loc[mask])
        return out

    return predict, report


def evaluate_year(
    df: pd.DataFrame,
    test_year: int,
    validation_years: int,
    apex_plus_factor: float,
    min_train_rows: int,
    feature_set: str,
) -> tuple[dict, dict]:
    train_for_shrink_end = test_year - validation_years - 1
    valid_start = test_year - validation_years
    valid_end = test_year - 1

    train_for_shrink_raw = df[df["Year"] <= train_for_shrink_end].copy()
    valid_raw = df[(df["Year"] >= valid_start) & (df["Year"] <= valid_end)].copy()
    final_train_raw = df[df["Year"] < test_year].copy()
    test_raw = df[df["Year"] == test_year].copy()

    if train_for_shrink_raw.empty or valid_raw.empty or final_train_raw.empty or test_raw.empty:
        raise ValueError(f"Not enough data to evaluate {test_year}")

    feats = model_features_for(feature_set)

    train_for_shrink, valid = prepare_fold(train_for_shrink_raw, valid_raw)
    base_for_shrink, _, _ = make_baseline(train_for_shrink)
    resid_for_shrink, fit_report = fit_position_residuals(train_for_shrink, base_for_shrink, feats=feats, min_train_rows=min_train_rows)
    shrink = tune_position_shrinkage(valid, base_for_shrink, resid_for_shrink)

    final_train, test = prepare_fold(final_train_raw, test_raw)
    pick_only, _ = make_pick_baseline(final_train)
    base, _, _ = make_baseline(final_train)
    resid, fit_report = fit_position_residuals(final_train, base, feats=feats, min_train_rows=min_train_rows)

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["market"] = base(scored)
    scored["position_apex_raw"] = score_apex(scored, base, resid, shrink)
    scored["position_apex_plus"] = score_apex_plus(scored["market"].to_numpy(), scored["position_apex_raw"].to_numpy(), apex_plus_factor)

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "validation_years": f"{valid_start}-{valid_end}",
        "feature_set": feature_set,
        "apex_plus_factor": apex_plus_factor,
        "min_train_rows": min_train_rows,
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
    }
    row.update(flatten_metrics("pick_only", metric_row(scored, "pick_only")))
    row.update(flatten_metrics("market", metric_row(scored, "market")))
    row.update(flatten_metrics("position_apex_raw", metric_row(scored, "position_apex_raw")))
    row.update(flatten_metrics("position_apex_plus", metric_row(scored, "position_apex_plus")))
    row["delta_plus_vs_pick_spearman_drafted"] = row["position_apex_plus_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_plus_vs_market_spearman_drafted"] = row["position_apex_plus_spearman_drafted"] - row["market_spearman_drafted"]
    row["delta_raw_vs_pick_spearman_drafted"] = row["position_apex_raw_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_plus_vs_raw_spearman_drafted"] = row["position_apex_plus_spearman_drafted"] - row["position_apex_raw_spearman_drafted"]
    return row, fit_report


def run_backtest(
    first_test_year: int,
    last_test_year: int,
    validation_years: int,
    data_dir: str | None,
    enriched_path: str | None,
    apex_plus_factor: float,
    min_train_rows: int,
    end_year: int | None = None,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> tuple[pd.DataFrame, dict]:
    effective_end_year = end_year if end_year is not None else max(last_test_year, 2016)
    df = load_modeling_table(data_dir=data_dir, enriched_path=enriched_path, end_year=effective_end_year)
    rows = []
    fit_reports = {}
    for year in range(first_test_year, last_test_year + 1):
        try:
            row, fit_report = evaluate_year(df, year, validation_years, apex_plus_factor, min_train_rows, feature_set)
        except ValueError as exc:
            print(f"Skipping {year}: {exc}")
            continue
        rows.append(row)
        fit_reports[str(year)] = fit_report
        print(
            f"{year}: position feature_set={feature_set} raw={row['position_apex_raw_spearman_drafted']:.3f} "
            f"position APEX+={row['position_apex_plus_spearman_drafted']:.3f} pick={row['pick_only_spearman_drafted']:.3f} "
            f"raw_delta={row['delta_raw_vs_pick_spearman_drafted']:.3f} plus_delta={row['delta_plus_vs_pick_spearman_drafted']:.3f}"
        )

    summary = pd.DataFrame(rows)
    report = aggregate_report(summary, first_test_year, last_test_year, validation_years, apex_plus_factor, effective_end_year, feature_set) if not summary.empty else {}
    report["model_type"] = "position_specific_residuals"
    report["feature_set"] = feature_set
    report["promotion_status"] = "top challenger; not headline unless it passes strict promotion gates"
    report["fit_reports_by_year"] = fit_reports
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=None, help="Last source-data year to load. Defaults to max(last-test-year, 2016).")
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--apex-plus-factor", type=float, default=DEFAULT_APEX_PLUS_FACTOR)
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET, choices=sorted(FEATURE_SETS))
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--features", type=str, default=None, help="Optional path to data/model_features.csv")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    summary, report = run_backtest(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        validation_years=args.validation_years,
        data_dir=args.data_dir,
        enriched_path=args.features,
        apex_plus_factor=args.apex_plus_factor,
        min_train_rows=args.min_train_rows,
        end_year=args.end_year,
        feature_set=args.feature_set,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "position_model_backtest_summary.csv", index=False)
    (out_dir / "position_model_backtest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
