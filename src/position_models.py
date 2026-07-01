"""Position-specific residual model backtest.

This script tests whether replacing one global residual model with separate
position-family residual models improves rolling out-of-time accuracy.

It can use optional production features when present in data/model_features.csv.
Build that file with:

    python src/build_features.py --write-templates

Then run:

    python src/position_models.py --first-test-year 2011 --last-test-year 2021
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
from pipeline import (
    CATS,
    FEATS_A,
    ROOT,
    load_dataset,
    make_baseline,
    make_pick_baseline,
    make_resid,
    metric_row,
    prepare_fold,
    safe_spearman,
    score_apex,
    tune_position_shrinkage,
)


def load_modeling_table(data_dir: str | None = None, enriched_path: str | None = None) -> pd.DataFrame:
    path = Path(enriched_path) if enriched_path else ENRICHED_FEATURE_FILE
    if path.exists():
        return pd.read_csv(path)
    base = load_dataset(data_dir=data_dir)
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
    min_train_rows: int,
    min_coverage: float,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    global_resid, _ = make_resid(train, base, feats=FEATS_A)
    models = {}
    report = {"families": {}, "fallback": "global_profile_residual"}

    for family, positions in POSITION_MODEL_GROUPS.items():
        mask = train["pos_g"].astype(str).isin(positions)
        subset = train.loc[mask].copy()
        extra = available_extra_features(subset, positions, min_coverage=min_coverage)
        feats = FEATS_A + extra
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
            "extra_features": extra,
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
    min_coverage: float,
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

    train_for_shrink, valid = prepare_fold(train_for_shrink_raw, valid_raw)
    base_for_shrink, _, _ = make_baseline(train_for_shrink)
    resid_for_shrink, fit_report = fit_position_residuals(
        train_for_shrink,
        base_for_shrink,
        min_train_rows=min_train_rows,
        min_coverage=min_coverage,
    )
    shrink = tune_position_shrinkage(valid, base_for_shrink, resid_for_shrink)

    final_train, test = prepare_fold(final_train_raw, test_raw)
    pick_only, _ = make_pick_baseline(final_train)
    base, _, _ = make_baseline(final_train)
    resid, fit_report = fit_position_residuals(
        final_train,
        base,
        min_train_rows=min_train_rows,
        min_coverage=min_coverage,
    )

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["market"] = base(scored)
    scored["position_apex_raw"] = score_apex(scored, base, resid, shrink)
    scored["position_apex_plus"] = score_apex_plus(
        scored["market"].to_numpy(),
        scored["position_apex_raw"].to_numpy(),
        apex_plus_factor,
    )

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "validation_years": f"{valid_start}-{valid_end}",
        "apex_plus_factor": apex_plus_factor,
        "min_train_rows": min_train_rows,
        "min_coverage": min_coverage,
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
    return row, fit_report


def run_backtest(
    first_test_year: int,
    last_test_year: int,
    validation_years: int,
    data_dir: str | None,
    enriched_path: str | None,
    apex_plus_factor: float,
    min_train_rows: int,
    min_coverage: float,
) -> tuple[pd.DataFrame, dict]:
    df = load_modeling_table(data_dir=data_dir, enriched_path=enriched_path)
    rows = []
    fit_reports = {}
    for year in range(first_test_year, last_test_year + 1):
        try:
            row, fit_report = evaluate_year(
                df,
                year,
                validation_years,
                apex_plus_factor,
                min_train_rows,
                min_coverage,
            )
        except ValueError as exc:
            print(f"Skipping {year}: {exc}")
            continue
        rows.append(row)
        fit_reports[str(year)] = fit_report
        print(
            f"{year}: position APEX+={row['position_apex_plus_spearman_drafted']:.3f} "
            f"vs pick={row['pick_only_spearman_drafted']:.3f} "
            f"delta={row['delta_plus_vs_pick_spearman_drafted']:.3f}"
        )

    summary = pd.DataFrame(rows)
    report = aggregate_report(summary, first_test_year, last_test_year, validation_years, apex_plus_factor) if not summary.empty else {}
    report["model_type"] = "position_specific_residuals"
    report["fit_reports_by_year"] = fit_reports
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--apex-plus-factor", type=float, default=DEFAULT_APEX_PLUS_FACTOR)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--min-coverage", type=float, default=0.15)
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
        min_coverage=args.min_coverage,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "position_model_backtest_summary.csv", index=False)
    (out_dir / "position_model_backtest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
