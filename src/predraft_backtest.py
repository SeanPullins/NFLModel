"""Pre-draft APEX backtest.

This evaluates whether the model can forecast before actual draft slot is known.
It requires consensus/expected-pick features, ideally built with:

    python src/build_features.py --write-templates

Minimum useful columns in data/consensus/consensus_board.csv:
    Year, Player, expected_pick or consensus_rank

The pre-draft market baseline uses expected_pick first, then consensus_rank, then
mock_avg_pick. Actual Pick is retained only for drafted-player filtering and final
comparison; it is not used as a feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from backtest import DEFAULT_APEX_PLUS_FACTOR, aggregate_report, flatten_metrics, score_apex_plus
from build_features import merge_optional_features
from feature_registry import ENRICHED_FEATURE_FILE, consensus_market_features
from pipeline import (
    FEATS_A,
    ROOT,
    load_dataset,
    make_resid,
    metric_row,
    prepare_fold,
)

PRE_DRAFT_MARKET_CANDIDATES = ["expected_pick", "consensus_rank", "mock_avg_pick"]


def load_modeling_table(
    data_dir: str | None = None,
    enriched_path: str | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    path = Path(enriched_path) if enriched_path else ENRICHED_FEATURE_FILE
    if path.exists():
        df = pd.read_csv(path)
        if end_year is not None:
            max_year = int(pd.to_numeric(df["Year"], errors="coerce").max())
            if max_year < end_year:
                raise ValueError(
                    f"{path} only covers through {max_year} but end_year={end_year}. "
                    "Rebuild it with: python src/build_features.py --end-year "
                    f"{end_year}"
                )
            df = df[pd.to_numeric(df["Year"], errors="coerce") <= end_year]
        return df
    base = load_dataset(data_dir=data_dir, end_year=end_year or 2016)
    enriched, _ = merge_optional_features(base)
    return enriched


def add_predraft_market(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["predraft_market_pick"] = np.nan
    for col in PRE_DRAFT_MARKET_CANDIDATES:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            out["predraft_market_pick"] = out["predraft_market_pick"].fillna(values)
    out["predraft_market_pick"] = out["predraft_market_pick"].clip(1, 300)
    return out


def make_predraft_market_baseline(train: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], np.ndarray], IsotonicRegression]:
    fit = train[train["predraft_market_pick"].notna() & train["y"].notna()].copy()
    if len(fit) < 100:
        raise ValueError("Not enough consensus/expected-pick rows to fit pre-draft market baseline")
    iso = IsotonicRegression(out_of_bounds="clip").fit(-fit["predraft_market_pick"], fit["y"])

    def predict(part: pd.DataFrame) -> np.ndarray:
        out = np.full(len(part), np.nan, dtype=float)
        mask = part["predraft_market_pick"].notna()
        if mask.any():
            out[mask.to_numpy()] = iso.predict(-part.loc[mask, "predraft_market_pick"])
        return out

    return predict, iso


def available_consensus_features(train: pd.DataFrame, min_coverage: float = 0.20) -> list[str]:
    """Use consensus context features, but keep the market proxy inside the baseline."""
    feats = []
    for col in consensus_market_features():
        if col in PRE_DRAFT_MARKET_CANDIDATES:
            continue
        if col in train.columns and pd.to_numeric(train[col], errors="coerce").notna().mean() >= min_coverage:
            feats.append(col)
    return feats


def evaluate_test_year(
    df: pd.DataFrame,
    test_year: int,
    validation_years: int,
    apex_plus_factor: float,
    min_coverage: float,
) -> dict:
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
    train_for_shrink = add_predraft_market(train_for_shrink)
    valid = add_predraft_market(valid)
    feats = FEATS_A + available_consensus_features(train_for_shrink, min_coverage=min_coverage)
    pre_market_for_shrink, _ = make_predraft_market_baseline(train_for_shrink)
    resid_for_shrink, _ = make_resid(train_for_shrink, pre_market_for_shrink, feats=feats)

    from pipeline import score_apex, tune_position_shrinkage

    shrink = tune_position_shrinkage(valid, pre_market_for_shrink, resid_for_shrink)

    final_train, test = prepare_fold(final_train_raw, test_raw)
    final_train = add_predraft_market(final_train)
    test = add_predraft_market(test)
    feats = FEATS_A + available_consensus_features(final_train, min_coverage=min_coverage)
    pre_market, _ = make_predraft_market_baseline(final_train)
    resid, _ = make_resid(final_train, pre_market, feats=feats)

    scored = test.copy()
    scored["predraft_market"] = pre_market(scored)
    scored["predraft_apex_raw"] = score_apex(scored, pre_market, resid, shrink)
    scored["predraft_apex_plus"] = score_apex_plus(
        scored["predraft_market"].to_numpy(),
        scored["predraft_apex_raw"].to_numpy(),
        apex_plus_factor,
    )

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "validation_years": f"{valid_start}-{valid_end}",
        "apex_plus_factor": apex_plus_factor,
        "features_used": "|".join(feats),
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
        "n_with_predraft_market": int(scored["predraft_market"].notna().sum()),
    }
    row.update(flatten_metrics("predraft_market", metric_row(scored, "predraft_market")))
    row.update(flatten_metrics("predraft_apex_raw", metric_row(scored, "predraft_apex_raw")))
    row.update(flatten_metrics("predraft_apex_plus", metric_row(scored, "predraft_apex_plus")))
    row["delta_plus_vs_market_spearman_drafted"] = row["predraft_apex_plus_spearman_drafted"] - row["predraft_market_spearman_drafted"]
    row["delta_raw_vs_market_spearman_drafted"] = row["predraft_apex_raw_spearman_drafted"] - row["predraft_market_spearman_drafted"]
    row["delta_plus_vs_raw_spearman_drafted"] = row["predraft_apex_plus_spearman_drafted"] - row["predraft_apex_raw_spearman_drafted"]
    return row


def run_backtest(
    first_test_year: int,
    last_test_year: int,
    validation_years: int,
    data_dir: str | None,
    enriched_path: str | None,
    apex_plus_factor: float,
    min_coverage: float,
    end_year: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = add_predraft_market(load_modeling_table(data_dir=data_dir, enriched_path=enriched_path, end_year=end_year))
    rows = []
    for year in range(first_test_year, last_test_year + 1):
        try:
            row = evaluate_test_year(df, year, validation_years, apex_plus_factor, min_coverage)
        except ValueError as exc:
            print(f"Skipping {year}: {exc}")
            continue
        rows.append(row)
        print(
            f"{year}: pre-draft APEX+={row['predraft_apex_plus_spearman_drafted']:.3f} "
            f"vs market={row['predraft_market_spearman_drafted']:.3f} "
            f"delta={row['delta_plus_vs_market_spearman_drafted']:.3f}"
        )
    summary = pd.DataFrame(rows)

    report_input = summary.rename(
        columns={
            "delta_plus_vs_market_spearman_drafted": "delta_plus_vs_pick_spearman_drafted",
            "delta_raw_vs_market_spearman_drafted": "delta_raw_vs_pick_spearman_drafted",
        }
    )
    report = aggregate_report(
        report_input,
        first_test_year,
        last_test_year,
        validation_years,
        apex_plus_factor,
    ) if not summary.empty else {}
    report["model_type"] = "predraft"
    report["market_proxy_order"] = PRE_DRAFT_MARKET_CANDIDATES
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--apex-plus-factor", type=float, default=DEFAULT_APEX_PLUS_FACTOR)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--end-year", type=int, default=None, help="Last source-data year to allow in the modeling table.")
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
        min_coverage=args.min_coverage,
        end_year=args.end_year,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "predraft_backtest_summary.csv", index=False)
    (out_dir / "predraft_backtest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
