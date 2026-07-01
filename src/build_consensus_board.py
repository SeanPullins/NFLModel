"""Build data/consensus/consensus_board.csv from public ESPN pre-draft rankings.

Source:
    JackLich10/nfl-draft-data, nfl_draft_prospects.csv

That file contains ESPN's pre-draft prospect board: overall rank (ovr_rk),
position rank (pos_rk), and scouting grade (grade) for draft classes since 2004.
These are pre-draft evaluations, so they can be used as a true pre-draft market
proxy without seeing actual draft-night picks.

Output columns follow the consensus template used by src/build_features.py and
src/predraft_backtest.py:

    Year, Player, Pos, College, consensus_rank, espn_grade, espn_pos_rank, source

`consensus_rank` (ESPN overall rank) is the pre-draft market proxy.
`espn_grade` and `espn_pos_rank` are additional pre-draft context features.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from download_source_data import download_text
from pipeline import ROOT

ESPN_PROSPECTS_URL = "https://raw.githubusercontent.com/JackLich10/nfl-draft-data/main/nfl_draft_prospects.csv"
OUT_PATH = ROOT / "data" / "consensus" / "consensus_board.csv"


def build_board(prospects: pd.DataFrame, min_year: int = 2004) -> pd.DataFrame:
    df = prospects.copy()
    df["Year"] = pd.to_numeric(df["draft_year"], errors="coerce")
    df["consensus_rank"] = pd.to_numeric(df["ovr_rk"], errors="coerce")
    df["espn_pos_rank"] = pd.to_numeric(df["pos_rk"], errors="coerce")
    df["espn_grade"] = pd.to_numeric(df["grade"], errors="coerce")

    df = df[df["Year"].ge(min_year) & df["player_name"].notna()].copy()
    # Keep rows that carry at least one pre-draft evaluation signal.
    df = df[df["consensus_rank"].notna() | df["espn_grade"].notna()].copy()
    df["Year"] = df["Year"].astype(int)

    out = pd.DataFrame(
        {
            "Year": df["Year"],
            "Player": df["player_name"].astype(str).str.strip(),
            "Pos": df.get("pos_abbr", "").fillna("").astype(str).str.upper(),
            "College": df.get("school", "Unknown").fillna("Unknown").astype(str),
            "consensus_rank": df["consensus_rank"],
            "espn_grade": df["espn_grade"],
            "espn_pos_rank": df["espn_pos_rank"],
            "source": "espn_prospects_jacklich10",
        }
    )
    out = out.sort_values(["Year", "consensus_rank"], na_position="last")
    return out.drop_duplicates(["Year", "Player"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    parser.add_argument("--min-year", type=int, default=2004)
    args = parser.parse_args()

    print(f"Downloading {ESPN_PROSPECTS_URL}")
    text = download_text(ESPN_PROSPECTS_URL)
    prospects = pd.read_csv(io.StringIO(text), low_memory=False)
    board = build_board(prospects, min_year=args.min_year)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.round(4).to_csv(out_path, index=False)

    coverage = (
        board.groupby("Year")
        .agg(
            n=("Player", "size"),
            with_rank=("consensus_rank", lambda s: int(s.notna().sum())),
            with_grade=("espn_grade", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    print(f"Wrote {out_path} rows={len(board):,}")
    print(json.dumps({str(r.Year): {"n": int(r.n), "with_rank": int(r.with_rank), "with_grade": int(r.with_grade)} for r in coverage.itertuples()}, indent=2))


if __name__ == "__main__":
    main()
