"""Nested validation for APEX Conservative factors.

This uses the existing factor-sweep by-year report. For each test year after a
minimum history window, it chooses the best factor using only prior test years,
then applies that selected factor to the current held-out year.

This is stricter than choosing a factor after seeing the whole validation span.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import ROOT
from validation_gates import evaluate_gates

DEFAULT_FACTOR_COL = "factor"
DEFAULT_DELTA_COL = "delta_plus_vs_pick_spearman_drafted"
DEFAULT_SCORE_COL = "apex_plus_spearman_drafted"


def choose_factor(history: pd.DataFrame, allowed_factors: set[float] | None = None) -> dict:
    data = history.copy()
    if allowed_factors is not None:
        data = data[data[DEFAULT_FACTOR_COL].round(4).isin(allowed_factors)].copy()
    rows = []
    for factor, group in data.groupby(DEFAULT_FACTOR_COL):
        delta = pd.to_numeric(group[DEFAULT_DELTA_COL], errors="coerce").dropna()
        if delta.empty:
            continue
        rows.append(
            {
                "factor": float(factor),
                "mean_lift": float(delta.mean()),
                "median_lift": float(delta.median()),
                "win_rate": float((delta > 0).mean()),
                "worst_window": float(delta.min()),
                "n_history": int(len(delta)),
            }
        )
    if not rows:
        raise ValueError("No usable factor history")
    ranked = pd.DataFrame(rows).sort_values(
        ["mean_lift", "median_lift", "win_rate", "worst_window"],
        ascending=[False, False, False, False],
    )
    return ranked.iloc[0].to_dict()


def run_nested(
    by_year: pd.DataFrame,
    min_history_years: int,
    allowed_factors: set[float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    years = sorted(pd.to_numeric(by_year["test_year"], errors="coerce").dropna().astype(int).unique())
    for year in years:
        history_years = [y for y in years if y < year]
        if len(history_years) < min_history_years:
            continue
        history = by_year[by_year["test_year"].isin(history_years)].copy()
        selected = choose_factor(history, allowed_factors=allowed_factors)
        factor = round(float(selected["factor"]), 4)
        current = by_year[(by_year["test_year"] == year) & (by_year[DEFAULT_FACTOR_COL].round(4) == factor)].copy()
        if current.empty:
            continue
        current_row = current.iloc[0].to_dict()
        rows.append(
            {
                "test_year": int(year),
                "selected_factor": factor,
                "history_years": f"{min(history_years)}-{max(history_years)}",
                "history_mean_lift": selected["mean_lift"],
                "history_median_lift": selected["median_lift"],
                "history_win_rate": selected["win_rate"],
                "history_worst_window": selected["worst_window"],
                "pick_only_spearman_drafted": current_row.get("pick_only_spearman_drafted"),
                "market_spearman_drafted": current_row.get("market_spearman_drafted"),
                "raw_apex_spearman_drafted": current_row.get("apex_raw_spearman_drafted"),
                "nested_factor_spearman_drafted": current_row.get(DEFAULT_SCORE_COL),
                "delta_nested_vs_pick_spearman_drafted": current_row.get(DEFAULT_DELTA_COL),
                "delta_raw_vs_pick_spearman_drafted": current_row.get("delta_raw_vs_pick_spearman_drafted"),
                "delta_nested_vs_raw_spearman_drafted": current_row.get("delta_plus_vs_raw_spearman_drafted"),
            }
        )
    summary = pd.DataFrame(rows)
    gate_report = evaluate_gates(summary, delta_col="delta_nested_vs_pick_spearman_drafted") if not summary.empty else {"pass": False, "reason": "No nested rows"}
    raw_gate_report = evaluate_gates(summary, delta_col="delta_raw_vs_pick_spearman_drafted") if not summary.empty else {"pass": False, "reason": "No nested rows"}
    report = {
        "method": "choose best factor using prior test years only, then apply to current year",
        "min_history_years": min_history_years,
        "allowed_factors": sorted(allowed_factors) if allowed_factors is not None else None,
        "years_evaluated": int(len(summary)),
        "nested_gate_report": gate_report,
        "raw_reference_gate_report_on_same_years": raw_gate_report,
        "selected_factor_counts": summary["selected_factor"].value_counts().sort_index().to_dict() if not summary.empty else {},
    }
    return summary, report


def parse_allowed(text: str | None) -> set[float] | None:
    if not text:
        return None
    return {round(float(x.strip()), 4) for x in text.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-by-year", type=str, default=str(ROOT / "reports" / "apex_factor_sweep_by_year.csv"))
    parser.add_argument("--min-history-years", type=int, default=4)
    parser.add_argument("--allowed-factors", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    by_year = pd.read_csv(args.factor_by_year)
    allowed = parse_allowed(args.allowed_factors)
    summary, report = run_nested(by_year, min_history_years=args.min_history_years, allowed_factors=allowed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "nested_factor_backtest_summary.csv", index=False)
    (out_dir / "nested_factor_backtest_report.json").write_text(json.dumps(report, indent=2))

    print("=== Nested factor backtest ===")
    print(summary.round(4).to_string(index=False) if not summary.empty else "No rows")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
