"""Feature ablation backtests for APEX.

This script answers the key validation question:

    Which feature families actually add out-of-time signal?

It compares profile-only, NCAA production-only, profile+production, offense-only,
defense-only, and position-specific variants over the same rolling validation
window used by the main model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backtest import aggregate_report, flatten_metrics
from feature_registry import POSITION_MODEL_GROUPS
from pipeline import (
    ATHLETIC_FEATURES,
    COLLEGE_PRODUCTION_FEATURES,
    CATS,
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
from validation_gates import evaluate_gates

OFFENSIVE_PRODUCTION_FEATURES = [
    "college_games",
    "college_pass_yds_pg",
    "college_pass_td_pg",
    "college_pass_int_pg",
    "college_pass_cmp_pct",
    "college_pass_td_int_ratio",
    "college_rush_yds_pg",
    "college_rush_td_pg",
    "college_rec_yds_pg",
    "college_rec_td_pg",
    "college_offensive_yds_pg",
    "college_total_td_pg",
]

DEFENSIVE_PRODUCTION_FEATURES = [
    "college_games",
    "college_tackles_pg",
    "college_sacks_pg",
    "college_ints_pg",
    "college_fumbles_pg",
    "college_def_playmaking_pg",
]

BASE_FEATURES = ["age", "col_enc"]

ABLATIONS: dict[str, dict] = {
    "global_profile_only": {
        "raw_features": ATHLETIC_FEATURES,
        "position_specific": False,
        "description": "Combine/profile features only, plus age and college encoding.",
    },
    "global_profile_plus_all_production": {
        "raw_features": ATHLETIC_FEATURES + COLLEGE_PRODUCTION_FEATURES,
        "position_specific": False,
        "description": "Current main model: profile plus all NCAA production features.",
    },
    "global_production_only": {
        "raw_features": COLLEGE_PRODUCTION_FEATURES,
        "position_specific": False,
        "description": "All NCAA production features only, plus age and college encoding.",
    },
    "global_offensive_production_only": {
        "raw_features": OFFENSIVE_PRODUCTION_FEATURES,
        "position_specific": False,
        "description": "Offensive NCAA production features only, plus age and college encoding.",
    },
    "global_defensive_production_only": {
        "raw_features": DEFENSIVE_PRODUCTION_FEATURES,
        "position_specific": False,
        "description": "Defensive NCAA production features only, plus age and college encoding.",
    },
    "position_profile_only": {
        "raw_features": ATHLETIC_FEATURES,
        "position_specific": True,
        "description": "Position-family residual models using profile features only.",
    },
    "position_profile_plus_all_production": {
        "raw_features": ATHLETIC_FEATURES + COLLEGE_PRODUCTION_FEATURES,
        "position_specific": True,
        "description": "Position-family residual models using profile plus all NCAA production features.",
    },
}


def feature_columns(raw_features: list[str]) -> list[str]:
    return [f"{feature}_z" for feature in raw_features] + BASE_FEATURES


def position_family(pos_g: str) -> str:
    pos = str(pos_g)
    for family, positions in POSITION_MODEL_GROUPS.items():
        if pos in positions:
            return family
    return "OTHER"


def fit_position_residuals(
    train: pd.DataFrame,
    base: Callable[[pd.DataFrame], np.ndarray],
    feats: list[str],
    *,
    min_train_rows: int = 300,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], dict]:
    global_resid, _ = make_resid(train, base, feats=feats)
    models = {}
    report = {"families": {}, "fallback": "global_residual"}

    for family, positions in POSITION_MODEL_GROUPS.items():
        subset = train[train["pos_g"].astype(str).isin(positions)].copy()
        if len(subset) < min_train_rows:
            report["families"][family] = {
                "trained": False,
                "n_train": int(len(subset)),
                "reason": f"n_train < {min_train_rows}",
            }
            continue
        resid, _ = make_resid(subset, base, feats=feats)
        models[family] = resid
        report["families"][family] = {"trained": True, "n_train": int(len(subset))}

    def predict(part: pd.DataFrame) -> np.ndarray:
        out = global_resid(part)
        families = part["pos_g"].astype(str).map(position_family)
        for family, resid in models.items():
            mask = families.eq(family)
            if mask.any():
                out[mask.to_numpy()] = resid(part.loc[mask])
        return out

    return predict, report


def evaluate_test_year(
    df: pd.DataFrame,
    test_year: int,
    raw_features: list[str],
    *,
    validation_years: int,
    position_specific: bool,
    min_train_rows: int,
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

    feats = feature_columns(raw_features)

    train_for_shrink, valid = prepare_fold(train_for_shrink_raw, valid_raw)
    base_for_shrink, _, _ = make_baseline(train_for_shrink)
    if position_specific:
        resid_for_shrink, fit_report = fit_position_residuals(
            train_for_shrink,
            base_for_shrink,
            feats,
            min_train_rows=min_train_rows,
        )
    else:
        resid_for_shrink, _ = make_resid(train_for_shrink, base_for_shrink, feats=feats)
        fit_report = {"families": {}, "fallback": "global_residual"}
    shrink = tune_position_shrinkage(valid, base_for_shrink, resid_for_shrink)

    final_train, test = prepare_fold(final_train_raw, test_raw)
    pick_only, _ = make_pick_baseline(final_train)
    base, _, _ = make_baseline(final_train)
    if position_specific:
        resid, fit_report = fit_position_residuals(final_train, base, feats, min_train_rows=min_train_rows)
    else:
        resid, _ = make_resid(final_train, base, feats=feats)

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["market"] = base(scored)
    scored["candidate"] = score_apex(scored, base, resid, shrink)

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "validation_years": f"{valid_start}-{valid_end}",
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
        "position_specific": bool(position_specific),
        "n_raw_features": int(len(raw_features)),
    }
    row.update(flatten_metrics("pick_only", metric_row(scored, "pick_only")))
    row.update(flatten_metrics("market", metric_row(scored, "market")))
    row.update(flatten_metrics("candidate", metric_row(scored, "candidate")))
    row["delta_candidate_vs_pick_spearman_drafted"] = row["candidate_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_candidate_vs_market_spearman_drafted"] = row["candidate_spearman_drafted"] - row["market_spearman_drafted"]
    row["delta_market_vs_pick_spearman_drafted"] = row["market_spearman_drafted"] - row["pick_only_spearman_drafted"]
    return row, fit_report


def summarize_ablation(name: str, by_year: pd.DataFrame, metadata: dict) -> dict:
    gates = evaluate_gates(by_year, delta_col="delta_candidate_vs_pick_spearman_drafted")
    metric = pd.to_numeric(by_year["delta_candidate_vs_pick_spearman_drafted"], errors="coerce")
    candidate_spearman = pd.to_numeric(by_year["candidate_spearman_drafted"], errors="coerce")
    pick_spearman = pd.to_numeric(by_year["pick_only_spearman_drafted"], errors="coerce")
    market_spearman = pd.to_numeric(by_year["market_spearman_drafted"], errors="coerce")
    return {
        "ablation": name,
        "description": metadata["description"],
        "position_specific": bool(metadata["position_specific"]),
        "n_raw_features": int(len(metadata["raw_features"])),
        "years_evaluated": int(len(by_year)),
        "pick_only_mean_spearman": float(pick_spearman.mean()),
        "market_mean_spearman": float(market_spearman.mean()),
        "candidate_mean_spearman": float(candidate_spearman.mean()),
        "mean_lift": float(metric.mean()),
        "median_lift": float(metric.median()),
        "win_rate": float((metric > 0).mean()),
        "worst_window": float(metric.min()),
        "best_window": float(metric.max()),
        "gate_pass": bool(gates.get("pass")),
    }


def run_ablation(
    first_test_year: int,
    last_test_year: int,
    end_year: int,
    validation_years: int,
    data_dir: str | None,
    ablations: list[str],
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_dataset(data_dir=data_dir, end_year=end_year)
    all_rows = []
    summary_rows = []
    fit_reports = {}

    for name in ablations:
        if name not in ABLATIONS:
            raise KeyError(f"Unknown ablation '{name}'. Available: {', '.join(ABLATIONS)}")
        metadata = ABLATIONS[name]
        rows = []
        fit_reports[name] = {}
        print(f"\n=== Ablation: {name} ===")
        print(metadata["description"])
        for test_year in range(first_test_year, last_test_year + 1):
            row, fit_report = evaluate_test_year(
                df,
                test_year,
                metadata["raw_features"],
                validation_years=validation_years,
                position_specific=metadata["position_specific"],
                min_train_rows=min_train_rows,
            )
            row["ablation"] = name
            row["description"] = metadata["description"]
            rows.append(row)
            fit_reports[name][str(test_year)] = fit_report
            print(
                f"{test_year}: candidate={row['candidate_spearman_drafted']:.3f} "
                f"pick={row['pick_only_spearman_drafted']:.3f} "
                f"delta={row['delta_candidate_vs_pick_spearman_drafted']:.3f}"
            )
        by_year = pd.DataFrame(rows)
        all_rows.extend(rows)
        summary_rows.append(summarize_ablation(name, by_year, metadata))

    by_year_all = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_lift", "median_lift", "win_rate", "worst_window"],
        ascending=[False, False, False, False],
    )
    winner = summary.iloc[0].to_dict() if not summary.empty else None
    report = {
        "first_test_year": first_test_year,
        "last_test_year": last_test_year,
        "end_year": end_year,
        "validation_years": validation_years,
        "ablations_requested": ablations,
        "winner_by_mean_lift": winner,
        "feature_sets": {
            name: {
                "description": ABLATIONS[name]["description"],
                "position_specific": ABLATIONS[name]["position_specific"],
                "raw_features": ABLATIONS[name]["raw_features"],
                "model_features": feature_columns(ABLATIONS[name]["raw_features"]),
            }
            for name in ablations
        },
        "fit_reports": fit_reports,
    }
    return summary, by_year_all, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--ablations", type=str, default=",".join(ABLATIONS.keys()))
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    ablations = [item.strip() for item in args.ablations.split(",") if item.strip()]
    summary, by_year, report = run_ablation(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        end_year=args.end_year,
        validation_years=args.validation_years,
        data_dir=args.data_dir,
        ablations=ablations,
        min_train_rows=args.min_train_rows,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "feature_ablation_summary.csv", index=False)
    by_year.round(4).to_csv(out_dir / "feature_ablation_by_year.csv", index=False)
    (out_dir / "feature_ablation_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== Feature ablation summary ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Winner by mean lift ===")
    print(json.dumps(report["winner_by_mean_lift"], indent=2))


if __name__ == "__main__":
    main()
