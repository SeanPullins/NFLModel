"""Compare candidate residual feature sets before promoting a model change.

This is intentionally separate from the production backtest. Use it to test whether
adding draft-capital interactions to the residual model improves out-of-time lift.

Examples:
    python src/experiment_feature_sets.py --first-test-year 2011 --last-test-year 2016
    APEX_DATA_DIR=/path/to/raw python src/experiment_feature_sets.py --last-test-year 2021
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import DEFAULT_APEX_PLUS_FACTOR, aggregate_report, flatten_metrics, score_apex_plus
from pipeline import (
    FEATS_A,
    FEATS_C,
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

FEATURE_SETS = {
    "profile_only": FEATS_A,
    "postdraft_interactions": FEATS_C,
}


def evaluate_year_feature_set(
    df: pd.DataFrame,
    test_year: int,
    feature_set_name: str,
    validation_years: int,
    apex_plus_factor: float,
) -> dict:
    feats = FEATURE_SETS[feature_set_name]
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
    resid_for_shrink, _ = make_resid(train_for_shrink, base_for_shrink, feats=feats)
    shrink = tune_position_shrinkage(valid, base_for_shrink, resid_for_shrink)

    final_train, test = prepare_fold(final_train_raw, test_raw)
    pick_only, _ = make_pick_baseline(final_train)
    base, _, _ = make_baseline(final_train)
    resid, _ = make_resid(final_train, base, feats=feats)

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["market"] = base(scored)
    scored["apex_raw"] = score_apex(scored, base, resid, shrink)
    scored["apex_plus"] = score_apex_plus(scored["market"].to_numpy(), scored["apex_raw"].to_numpy(), apex_plus_factor)

    row = {
        "feature_set": feature_set_name,
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "validation_years": f"{valid_start}-{valid_end}",
        "apex_plus_factor": apex_plus_factor,
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
    }
    row.update(flatten_metrics("pick_only", metric_row(scored, "pick_only")))
    row.update(flatten_metrics("market", metric_row(scored, "market")))
    row.update(flatten_metrics("apex_raw", metric_row(scored, "apex_raw")))
    row.update(flatten_metrics("apex_plus", metric_row(scored, "apex_plus")))
    row["delta_raw_vs_pick_spearman_drafted"] = row["apex_raw_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_plus_vs_pick_spearman_drafted"] = row["apex_plus_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_plus_vs_market_spearman_drafted"] = row["apex_plus_spearman_drafted"] - row["market_spearman_drafted"]
    row["delta_plus_vs_raw_spearman_drafted"] = row["apex_plus_spearman_drafted"] - row["apex_raw_spearman_drafted"]
    return row


def run_experiment(first_test_year: int, last_test_year: int, validation_years: int, data_dir: str | None, apex_plus_factor: float) -> tuple[pd.DataFrame, dict]:
    df = load_dataset(data_dir=data_dir)
    rows = []
    for feature_set_name in FEATURE_SETS:
        for test_year in range(first_test_year, last_test_year + 1):
            try:
                row = evaluate_year_feature_set(df, test_year, feature_set_name, validation_years, apex_plus_factor)
            except ValueError as exc:
                print(f"Skipping {feature_set_name} {test_year}: {exc}")
                continue
            rows.append(row)
            print(
                f"{feature_set_name} {test_year}: "
                f"APEX+={row['apex_plus_spearman_drafted']:.3f} "
                f"delta={row['delta_plus_vs_pick_spearman_drafted']:.3f}"
            )

    summary = pd.DataFrame(rows)
    report = {}
    if not summary.empty:
        for feature_set_name, group in summary.groupby("feature_set"):
            report[feature_set_name] = aggregate_report(group, first_test_year, last_test_year, validation_years, apex_plus_factor)

    if len(report) == 2:
        a = report["profile_only"]["apex_plus_vs_pick"]["mean"]
        b = report["postdraft_interactions"]["apex_plus_vs_pick"]["mean"]
        report["recommendation"] = {
            "winner_by_mean_delta": "postdraft_interactions" if b > a else "profile_only",
            "mean_delta_difference": float(b - a),
            "rule": "Promote postdraft_interactions only if it improves average and median lift without lowering win rate."
        }
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2016)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--apex-plus-factor", type=float, default=DEFAULT_APEX_PLUS_FACTOR)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    summary, report = run_experiment(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        validation_years=args.validation_years,
        data_dir=args.data_dir,
        apex_plus_factor=args.apex_plus_factor,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "feature_set_experiment_summary.csv", index=False)
    (out_dir / "feature_set_experiment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
