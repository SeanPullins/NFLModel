"""Accuracy evaluation of the published board.

Evaluates what the site actually shows - apex_conservative_050 (main grade),
raw APEX, the PFF-informed layer, fair slots, and tier odds - against real
career outcomes on mature classes, plus data-integrity checks.

Notes on honesty:
- The board model is fit on all data, so mature-class Spearman here is partly
  in-sample; the out-of-time truth lives in reports/rolling_backtest_summary.
  This report is about the *published numbers* being coherent and calibrated.
- Tier odds were calibrated in-sample by construction; the reliability table
  quantifies how far off they are anyway.

Usage:
    python src/evaluate_board.py [--mature-end 2021]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, safe_spearman

BOARD = ROOT / "data" / "apex_board.csv"
PFF_SCORES = ROOT / "data" / "pff_scores.csv"
OUT = ROOT / "reports" / "board_evaluation.json"


def load_board() -> pd.DataFrame:
    board = pd.read_csv(BOARD)
    if PFF_SCORES.exists():
        pff = pd.read_csv(PFF_SCORES)[["Year", "Player", "apex_pff", "pff_edge"]]
        board = board.merge(pff.drop_duplicates(["Year", "Player"]), on=["Year", "Player"], how="left")
    return board


def score_accuracy(mature: pd.DataFrame) -> dict:
    drafted = mature[mature["Pick"].notna() & mature["Pick"].lt(263) & mature["y"].notna()].copy()
    out = {}
    for col in ["apex_conservative_050", "apex", "apex_pff", "exp_at_pick", "implied_pick"]:
        if col not in drafted.columns:
            continue
        score = -drafted[col] if col == "implied_pick" else drafted[col]
        out[col] = {
            "spearman_vs_outcome": round(safe_spearman(score, drafted["y"]), 4),
            "n": int(score.notna().sum()),
        }
    pick_rho = safe_spearman(-drafted["Pick"], drafted["y"])
    out["actual_pick_baseline"] = {"spearman_vs_outcome": round(pick_rho, 4), "n": int(len(drafted))}
    return out


def tier_calibration(mature: pd.DataFrame) -> list[dict]:
    drafted = mature[mature["Pick"].lt(263) & mature["y"].notna() & mature["p_star"].notna()].copy()
    drafted["star_actual"] = (drafted["y"] >= 0.85).astype(int)
    drafted["bust_actual"] = (drafted["y"] < 0.45).astype(int)
    rows = []
    drafted["bucket"] = pd.qcut(drafted["p_star"], q=8, duplicates="drop")
    for bucket, g in drafted.groupby("bucket", observed=True):
        rows.append({
            "p_star_bucket": str(bucket),
            "n": int(len(g)),
            "predicted_star": round(float(g["p_star"].mean()), 3),
            "actual_star": round(float(g["star_actual"].mean()), 3),
            "predicted_bust": round(float(g["p_bust"].mean()), 3),
            "actual_bust": round(float(g["bust_actual"].mean()), 3),
        })
    return rows


def fair_slot_quality(mature: pd.DataFrame) -> dict:
    drafted = mature[mature["Pick"].lt(263) & mature["y"].notna() & mature["pick_delta"].notna()].copy()
    steals = drafted[drafted["pick_delta"] >= 15]
    reaches = drafted[drafted["pick_delta"] <= -15]
    neutral = drafted[drafted["pick_delta"].abs() < 15]

    def beat_market(g: pd.DataFrame) -> float:
        return round(float((g["y"] > g["exp_at_pick"]).mean()), 3)

    return {
        "steal_calls": {"n": int(len(steals)), "beat_market_rate": beat_market(steals)},
        "neutral": {"n": int(len(neutral)), "beat_market_rate": beat_market(neutral)},
        "reach_calls": {"n": int(len(reaches)), "beat_market_rate": beat_market(reaches)},
    }


def integrity_checks(board: pd.DataFrame) -> dict:
    checks = {}
    for col in ["apex", "apex_conservative_050", "exp_at_pick", "apex_pff"]:
        if col in board.columns:
            vals = pd.to_numeric(board[col], errors="coerce")
            checks[f"{col}_out_of_range"] = int(((vals < 0) | (vals > 1)).sum())
            checks[f"{col}_missing"] = int(vals.isna().sum())
    drafted = board[board["Pick"].notna() & board["Pick"].lt(263)]
    per_year = drafted.groupby("Year").size()
    checks["drafted_counts_recent"] = {str(k): int(v) for k, v in per_year.loc[per_year.index >= 2022].items()}
    checks["duplicate_year_player"] = int(board.duplicated(["Year", "Player"]).sum())
    immature_with_outcome = board[(board["Year"] >= 2025) & board["y"].notna()]
    checks["immature_rows_showing_outcomes"] = int(len(immature_with_outcome))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mature-start", type=int, default=2011)
    parser.add_argument("--mature-end", type=int, default=2021)
    args = parser.parse_args()

    board = load_board()
    mature = board[board["Year"].between(args.mature_start, args.mature_end)].copy()

    report = {
        "scope": f"published board, mature classes {args.mature_start}-{args.mature_end}",
        "caveat": "board scores are fit on all data (partly in-sample); out-of-time lift lives in rolling_backtest_summary",
        "score_accuracy": score_accuracy(mature),
        "tier_calibration": tier_calibration(mature),
        "fair_slot_quality": fair_slot_quality(mature),
        "integrity": integrity_checks(board),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
