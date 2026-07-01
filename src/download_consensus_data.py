"""Download real pre-draft consensus data for true pre-draft forecasting.

This closes the "true pre-draft forecasting" gap tracked in
src/predraft_backtest.py: instead of a hand-written template, this pulls
JackLich10/nfl-draft-data's ESPN draft-prospect table, which carries ESPN's
pre-draft overall rank, position rank, and grade for classes back to 2004
(actual draft outcomes are excluded from the feature set on purpose - only
pre-draft evaluation columns are kept).

Writes:
    data/consensus/consensus_board.csv

Columns:
    Year, Player, consensus_rank (ESPN pre-draft overall rank),
    espn_grade (ESPN pre-draft player grade), espn_pos_rk (ESPN pre-draft
    position rank)

Honest result from the 2011-2021 rolling backtest (see docs/VALIDATION.md):
this consensus data does NOT beat the actual draft market as a forecasting
input, and does not meaningfully improve the post-draft profile model either.
It is kept available for ablations and future experiments, not as a promoted
headline feature.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from pipeline import ROOT, norm

ESPN_PROSPECTS_URL = "https://raw.githubusercontent.com/JackLich10/nfl-draft-data/master/nfl_draft_prospects.csv"


def download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "APEX-NFLModel/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def build_consensus_board(raw_text: str) -> pd.DataFrame:
    espn = pd.read_csv(pd.io.common.StringIO(raw_text))
    espn = espn.rename(columns={"draft_year": "Year", "player_name": "Player"})
    espn["Year"] = pd.to_numeric(espn["Year"], errors="coerce")
    espn = espn[espn["Year"].notna() & espn["Player"].notna()].copy()
    espn["Year"] = espn["Year"].astype(int)
    espn["key"] = espn["Player"].map(norm) + "_" + espn["Year"].astype(str)
    espn = espn.drop_duplicates("key")

    out = espn[["Year", "Player", "ovr_rk", "pos_rk", "grade"]].rename(
        columns={"ovr_rk": "consensus_rank", "pos_rk": "espn_pos_rk", "grade": "espn_grade"}
    )
    for col in ("consensus_rank", "espn_pos_rk", "espn_grade"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "consensus" / "consensus_board.csv"))
    args = parser.parse_args()

    print(f"Downloading {ESPN_PROSPECTS_URL}")
    try:
        raw_text = download_text(ESPN_PROSPECTS_URL)
    except Exception as exc:
        print(f"WARNING: consensus data download failed: {exc}")
        print("Leaving any existing data/consensus/consensus_board.csv untouched.")
        return

    board = build_consensus_board(raw_text)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.round(4).to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(board):,} year_range={int(board['Year'].min())}-{int(board['Year'].max())}")


if __name__ == "__main__":
    main()
