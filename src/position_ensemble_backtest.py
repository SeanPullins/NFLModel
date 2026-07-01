"""Position-weighted ensemble backtest.

Tests whether a controlled blend of global profile-only raw APEX and
position-specific profile-only raw APEX can improve accuracy while reducing the
risk of the pure position-specific model.

For each test year:
1. Fit global and position residual models on pre-validation data.
2. Choose an ensemble weight using only the rolling validation fold.
3. Refit on all pre-test data and score the held-out test year.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import aggregate_report, flatten_metrics
from feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS, model_features_for
from pipeline import (
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
from position_models import fit_position_residuals
from validation_gates import evaluate_gates


def parse_weights(text: str) -> list[float]:
    return sorted({round(float(x.strip()), 4) for x in text.split(",") if x.strip()})


def choose_weight(valid: pd.DataFrame, global_score: np.ndarray, position_score: np.ndarray, weights: list[float]) -> dict:
    drafted = valid[valid["Pick"].lt(263)].copy()
    mask = valid["Pick"].lt(263).to_numpy()
    rows = []
    for weight in weights:
        score = (1 - weight) * global_score + weight * position_score
        metric = safe_spearman(score[mask], drafted["y"])
        rows.append({"weight": float(weight), "validation_spearman": metric})
    ranked = pd.DataFrame(rows).sort_values(["validation_spearman", "weight"], ascending=[False, True])
    return ranked.iloc[0].to_dict()


def evaluate_year(
    df: pd.DataFrame,
    test_year: int,
    validation_years: int,
    feature_set: str,
    weights: list[float],
    min_train_rows: int,
) -> tuple[dict, dict]:
    feats = model_features_for(feature_set)
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
    global_resid_for_shrink, _ = make_resid(train_for_shrink, base_for_shrink, feats=feats)
    position_resid_for_shrink, pos_fit_for_shrink = fit_position_residuals(
        train_for_shrink,
        base_for_shrink,
        feats=feats,
        min_train_rows=min_train_rows,
    )
    global_shrink = tune_position_shrinkage(valid, base_for_shrink, global_resid_for_shrink)
    position_shrink = tune_position_shrinkage(valid, base_for_shrink, position_resid_for_shrink)
    valid_global = score_apex(valid, base_for_shrink, global_resid_for_shrink, global_shrink)
    valid_position = score_apex(valid, base_for_shrink, position_resid_for_shrink, position_shrink)
    selected = choose_weight(valid, valid_global, valid_position, weights)
    weight = float(selected["weight"])

    final_train, test = prepare_fold(final_train_raw, test_raw)
    pick_only, _ = make_pick_baseline(final_train)
    base, _, _ = make_baseline(final_train)
    global_resid, _ = make_resid(final_train, base, feats=feats)
    position_resid, pos_fit = fit_position_residuals(final_train, base, feats=feats, min_train_rows=min_train_rows)

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["market"] = base(scored)
    scored["global_profile_raw"] = score_apex(scored, base, global_resid, global_shrink)
    scored["position_profile_raw"] = score_apex(scored, base, position_resid, position_shrink)
    scored["position_weighted_ensemble"] = (1 - weight) * scored["global_profile_raw"] + weight * scored["position_profile_raw"]

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "validation_years": f"{valid_start}-{valid_end}",
        "feature_set": feature_set,
        "selected_position_weight": weight,
        "validation_spearman_for_weight": selected["validation_spearman"],
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
    }
    row.update(flatten_metrics("pick_only", metric_row(scored, "pick_only")))
    row.update(flatten_metrics("market", metric_row(scored, "market")))
    row.update(flatten_metrics("global_profile_raw", metric_row(scored, "global_profile_raw")))
    row.update(flatten_metrics("position_profile_raw", metric_row(scored, "position_profile_raw")))
    row.update(flatten_metrics("position_weighted_ensemble", metric_row(scored, "position_weighted_ensemble")))
    row["delta_ensemble_vs_pick_spearman_drafted"] = row["position_weighted_ensemble_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_global_vs_pick_spearman_drafted"] = row["global_profile_raw_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_position_vs_pick_spearman_drafted"] = row["position_profile_raw_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_ensemble_vs_global_spearman_drafted"] = row["position_weighted_ensemble_spearman_drafted"] - row["global_profile_raw_spearman_drafted"]
    return row, {"validation_position_fit": pos_fit_for_shrink, "final_position_fit": pos_fit}


def run_backtest(
    first_test_year: int,
    last_test_year: int,
    end_year: int,
    validation_years: int,
    feature_set: str,
    weights: list[float],
    min_train_rows: int,
    data_dir: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = load_dataset(data_dir=data_dir, end_year=end_year)
    rows = []
    fit_reports = {}
    for year in range(first_test_year, last_test_year + 1):
        try:
            row, fit_report = evaluate_year(df, year, validation_years, feature_set, weights, min_train_rows)
        except ValueError as exc:
            print(f"Skipping {year}: {exc}")
            continue
        rows.append(row)
        fit_reports[str(year)] = fit_report
        print(
            f"{year}: ensemble={row['position_weighted_ensemble_spearman_drafted']:.3f} "
            f"pick={row['pick_only_spearman_drafted']:.3f} "
            f"weight={row['selected_position_weight']:.2f} "
            f"delta={row['delta_ensemble_vs_pick_spearman_drafted']:.3f}"
        )
    summary = pd.DataFrame(rows)
    gate = evaluate_gates(summary, delta_col="delta_ensemble_vs_pick_spearman_drafted") if not summary.empty else {"pass": False, "reason": "No rows"}
    global_gate = evaluate_gates(summary, delta_col="delta_global_vs_pick_spearman_drafted") if not summary.empty else {"pass": False, "reason": "No rows"}
    position_gate = evaluate_gates(summary, delta_col="delta_position_vs_pick_spearman_drafted") if not summary.empty else {"pass": False, "reason": "No rows"}
    report = {
        "model_type": "position_weighted_ensemble",
        "feature_set": feature_set,
        "first_test_year": first_test_year,
        "last_test_year": last_test_year,
        "end_year": end_year,
        "validation_years": validation_years,
        "weights_tested": weights,
        "years_evaluated": int(len(summary)),
        "ensemble_gate_report": gate,
        "global_reference_gate_report": global_gate,
        "position_reference_gate_report": position_gate,
        "selected_weight_counts": summary["selected_position_weight"].value_counts().sort_index().to_dict() if not summary.empty else {},
        "fit_reports_by_year": fit_reports,
    }
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET, choices=sorted(FEATURE_SETS))
    parser.add_argument("--weights", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    summary, report = run_backtest(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        end_year=args.end_year,
        validation_years=args.validation_years,
        feature_set=args.feature_set,
        weights=weights,
        min_train_rows=args.min_train_rows,
        data_dir=args.data_dir,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "position_ensemble_backtest_summary.csv", index=False)
    (out_dir / "position_ensemble_backtest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
