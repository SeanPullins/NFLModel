"""Sweep APEX+ residual factors and promote only if validation gates pass.

APEX+ uses:

    market + factor * (raw_apex - market)

Factor 0.0 is the market baseline. Factor 1.0 is raw APEX. Factors above 1.0
amplify the residual. This script tests candidate factors and recommends a
promoted factor only if it passes promotion gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import DEFAULT_APEX_PLUS_FACTOR, run_backtest
from pipeline import ROOT
from validation_gates import evaluate_gates


def parse_factors(text: str | None, start: float, stop: float, step: float) -> list[float]:
    if text:
        return sorted({round(float(x.strip()), 4) for x in text.split(",") if x.strip()})
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(float(current), 4))
        current += step
    return values


def summarize_factor(factor: float, summary: pd.DataFrame, gates: dict | None = None) -> dict:
    gate_report = evaluate_gates(summary, delta_col="delta_plus_vs_pick_spearman_drafted", gates=gates)
    raw_gate_report = evaluate_gates(summary, delta_col="delta_raw_vs_pick_spearman_drafted", gates=gates)
    return {
        "factor": factor,
        "years_evaluated": int(len(summary)),
        "apex_plus_mean_lift": gate_report.get("checks", {}).get("mean_lift"),
        "apex_plus_median_lift": gate_report.get("checks", {}).get("median_lift"),
        "apex_plus_win_rate": gate_report.get("checks", {}).get("win_rate"),
        "apex_plus_worst_window": gate_report.get("checks", {}).get("worst_window"),
        "apex_plus_gate_pass": bool(gate_report.get("pass")),
        "raw_apex_mean_lift": raw_gate_report.get("checks", {}).get("mean_lift"),
        "raw_apex_median_lift": raw_gate_report.get("checks", {}).get("median_lift"),
        "raw_apex_win_rate": raw_gate_report.get("checks", {}).get("win_rate"),
        "raw_apex_worst_window": raw_gate_report.get("checks", {}).get("worst_window"),
    }


def choose_promotion(factor_summary: pd.DataFrame, gates: dict | None = None) -> dict:
    if factor_summary.empty:
        return {
            "promoted_factor": None,
            "headline_model": "raw_apex",
            "reason": "No factor results were produced.",
        }

    passing = factor_summary[factor_summary["apex_plus_gate_pass"]].copy()
    if passing.empty:
        best_raw = factor_summary.iloc[0]
        return {
            "promoted_factor": None,
            "headline_model": "raw_apex",
            "reason": "No APEX+ residual factor passed promotion gates. Use raw APEX as the honest headline until a factor passes.",
            "raw_apex_reference": {
                "mean_lift": float(best_raw["raw_apex_mean_lift"]),
                "median_lift": float(best_raw["raw_apex_median_lift"]),
                "win_rate": float(best_raw["raw_apex_win_rate"]),
                "worst_window": float(best_raw["raw_apex_worst_window"]),
            },
        }

    passing = passing.sort_values(
        ["apex_plus_mean_lift", "apex_plus_median_lift", "apex_plus_win_rate", "apex_plus_worst_window"],
        ascending=[False, False, False, False],
    )
    winner = passing.iloc[0]
    return {
        "promoted_factor": float(winner["factor"]),
        "headline_model": "apex_plus",
        "reason": "Highest mean lift among APEX+ factors passing promotion gates.",
        "winner": {
            "factor": float(winner["factor"]),
            "mean_lift": float(winner["apex_plus_mean_lift"]),
            "median_lift": float(winner["apex_plus_median_lift"]),
            "win_rate": float(winner["apex_plus_win_rate"]),
            "worst_window": float(winner["apex_plus_worst_window"]),
        },
    }


def run_sweep(
    first_test_year: int,
    last_test_year: int,
    end_year: int | None,
    validation_years: int,
    data_dir: str | None,
    factors: list[float],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    details = []
    reports = {}
    for factor in factors:
        print(f"\n=== APEX+ factor {factor} ===")
        summary, _, report = run_backtest(
            first_test_year=first_test_year,
            last_test_year=last_test_year,
            validation_years=validation_years,
            data_dir=data_dir,
            apex_plus_factor=factor,
            end_year=end_year,
        )
        if summary.empty:
            continue
        s = summary.copy()
        s["factor"] = factor
        details.append(s)
        rows.append(summarize_factor(factor, s))
        reports[str(factor)] = report

    factor_summary = pd.DataFrame(rows)
    detail_summary = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    promotion = choose_promotion(factor_summary)
    report = {
        "first_test_year": first_test_year,
        "last_test_year": last_test_year,
        "end_year": end_year,
        "validation_years": validation_years,
        "factors_tested": factors,
        "promotion": promotion,
        "factor_reports": reports,
    }
    out_dir.mkdir(exist_ok=True)
    factor_summary.round(4).to_csv(out_dir / "apex_factor_sweep_summary.csv", index=False)
    detail_summary.round(4).to_csv(out_dir / "apex_factor_sweep_by_year.csv", index=False)
    (out_dir / "apex_factor_sweep_report.json").write_text(json.dumps(report, indent=2))
    return factor_summary, detail_summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--factor-start", type=float, default=0.0)
    parser.add_argument("--factor-stop", type=float, default=3.5)
    parser.add_argument("--factor-step", type=float, default=0.25)
    parser.add_argument("--factors", type=str, default=None, help="Comma-separated factor list. Overrides start/stop/step.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    factors = parse_factors(args.factors, args.factor_start, args.factor_stop, args.factor_step)
    factor_summary, _, report = run_sweep(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        end_year=args.end_year,
        validation_years=args.validation_years,
        data_dir=args.data_dir,
        factors=factors,
        out_dir=Path(args.out_dir),
    )
    print("\n=== Factor summary ===")
    print(factor_summary.round(4).to_string(index=False))
    print("\n=== Promotion report ===")
    print(json.dumps(report["promotion"], indent=2))


if __name__ == "__main__":
    main()
