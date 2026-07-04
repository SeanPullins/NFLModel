"""Add conservative APEX blend scores and display-board fields.

APEX Conservative is not residual amplification. It is a safer blend between the
market baseline and raw APEX:

    conservative = market + factor * (raw_apex - market)

The 0.50 blend passed current stability gates in the 2011-2021 factor sweep, but
is kept as a candidate until nested validation confirms it can be selected using
only prior years.

This step also writes the plain display fields the site needs: actual pick,
should-have-gone slot, slot value, and calmer historical miss risk. That keeps
the public board from falling back to compressed old p_bust buckets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline import ROOT
try:
    from calibrate_outcome_odds import add_display_odds
except Exception:  # pragma: no cover
    add_display_odds = None

DEFAULT_FACTORS = [0.25, 0.50, 0.75]


def parse_factors(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_FACTORS
    return sorted({round(float(x.strip()), 4) for x in text.split(",") if x.strip()})


def factor_label(factor: float) -> str:
    return f"{int(round(factor * 100)):03d}"


def add_conservative_scores(board: pd.DataFrame, factors: list[float]) -> pd.DataFrame:
    out = board.copy()
    required = {"apex", "exp_at_pick"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Board is missing required columns: {sorted(missing)}")

    raw = pd.to_numeric(out["apex"], errors="coerce")
    market = pd.to_numeric(out["exp_at_pick"], errors="coerce")
    for factor in factors:
        label = factor_label(factor)
        score_col = f"apex_conservative_{label}"
        surplus_col = f"conservative_surplus_{label}"
        out[score_col] = (market + factor * (raw - market)).clip(0.01, 0.99)
        out[surplus_col] = out[score_col] - market
    out["recommended_candidate_score"] = out.get("apex_conservative_050", raw)
    out["recommended_candidate_status"] = "candidate_conservative_050_pending_nested_validation"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=str, default=str(ROOT / "data" / "apex_board.csv"))
    parser.add_argument("--factors", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.board)
    board = pd.read_csv(path)
    factors = parse_factors(args.factors)
    out = add_conservative_scores(board, factors)
    if add_display_odds is not None:
        out = add_display_odds(out)
    out.round(4).to_csv(path, index=False)
    print(f"Added conservative APEX scores and display slots to {path}: {', '.join(str(f) for f in factors)}")


if __name__ == "__main__":
    main()
