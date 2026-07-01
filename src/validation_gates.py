"""Promotion gates for APEX model experiments.

A candidate model should not become the public headline model just because one
metric improved in one window. These gates force average, median, win-rate, and
worst-window checks.

Default CLI gate now evaluates raw APEX lift because raw APEX is the public
headline model. APEX+ factor sweeps still pass their APEX+ delta explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_GATES = {
    "min_mean_lift": 0.0,
    "min_median_lift": 0.0,
    "min_win_rate": 0.60,
    "max_worst_window_loss": 0.02,
    "require_non_negative_precision32_delta": False,
    "require_non_negative_precision64_delta": False,
}

DEFAULT_RAW_DELTA_COL = "delta_raw_vs_pick_spearman_drafted"
DEFAULT_PLUS_DELTA_COL = "delta_plus_vs_pick_spearman_drafted"


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.Series(dtype=float)


def evaluate_gates(
    summary: pd.DataFrame,
    delta_col: str = DEFAULT_RAW_DELTA_COL,
    gates: dict | None = None,
) -> dict:
    gates = {**DEFAULT_GATES, **(gates or {})}
    delta = _series(summary, delta_col).dropna()
    if delta.empty:
        return {
            "pass": False,
            "reason": f"No usable values for {delta_col}",
            "gates": gates,
        }

    checks = {
        "mean_lift": float(delta.mean()),
        "median_lift": float(delta.median()),
        "win_rate": float((delta > 0).mean()),
        "worst_window": float(delta.min()),
        "n_windows": int(len(delta)),
    }

    failures: list[str] = []
    if checks["mean_lift"] < gates["min_mean_lift"]:
        failures.append(f"mean_lift {checks['mean_lift']:.4f} < {gates['min_mean_lift']:.4f}")
    if checks["median_lift"] < gates["min_median_lift"]:
        failures.append(f"median_lift {checks['median_lift']:.4f} < {gates['min_median_lift']:.4f}")
    if checks["win_rate"] < gates["min_win_rate"]:
        failures.append(f"win_rate {checks['win_rate']:.3f} < {gates['min_win_rate']:.3f}")
    if checks["worst_window"] < -abs(gates["max_worst_window_loss"]):
        failures.append(f"worst_window {checks['worst_window']:.4f} < -{abs(gates['max_worst_window_loss']):.4f}")

    if gates.get("require_non_negative_precision32_delta"):
        col = "delta_plus_vs_pick_precision_at_32"
        if col in summary.columns and _series(summary, col).dropna().mean() < 0:
            failures.append(f"{col} average is negative")

    if gates.get("require_non_negative_precision64_delta"):
        col = "delta_plus_vs_pick_precision_at_64"
        if col in summary.columns and _series(summary, col).dropna().mean() < 0:
            failures.append(f"{col} average is negative")

    return {
        "pass": not failures,
        "checks": checks,
        "failures": failures,
        "gates": gates,
        "delta_col": delta_col,
    }


def compare_candidates(
    summary: pd.DataFrame,
    candidate_col: str = "feature_set",
    delta_col: str = DEFAULT_RAW_DELTA_COL,
) -> dict:
    if candidate_col not in summary.columns:
        return {"error": f"Missing candidate column {candidate_col}"}
    out = {}
    for name, group in summary.groupby(candidate_col):
        out[str(name)] = evaluate_gates(group, delta_col=delta_col)
    passing = [name for name, result in out.items() if result.get("pass")]
    if passing:
        ranked = sorted(passing, key=lambda name: out[name]["checks"]["mean_lift"], reverse=True)
        out["recommendation"] = {"promote": ranked[0], "reason": "highest mean lift among candidates passing gates"}
    else:
        out["recommendation"] = {"promote": None, "reason": "no candidate passed the promotion gates"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv")
    parser.add_argument("--candidate-col", default=None)
    parser.add_argument("--delta-col", default=DEFAULT_RAW_DELTA_COL)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit with a non-zero status (failing the CI job) if the gate check does not pass. "
        "Default is report-only, since not every gate here backs a promoted model.",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if args.candidate_col:
        report = compare_candidates(summary, args.candidate_col, args.delta_col)
        passed = report.get("recommendation", {}).get("promote") is not None
    else:
        report = evaluate_gates(summary, args.delta_col)
        passed = bool(report.get("pass"))

    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)

    if args.enforce and not passed:
        print(f"GATE FAILED for {args.summary_csv} ({args.delta_col}) - failing build.")
        sys.exit(1)


if __name__ == "__main__":
    main()
