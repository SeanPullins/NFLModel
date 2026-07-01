"""Build data/production/qb_production.csv from public ESPN college QBR.

Sources (JackLich10/nfl-draft-data):
    college_qbr.csv          ESPN college Total QBR seasons, 2004-2020
    nfl_draft_prospects.csv  prospect list with ESPN guid per draft year

Only seasons strictly before a player's draft year are used, so every feature
is knowable pre-draft. NFL outcomes are never touched.

Features written (per QB, per draft year):
    qb_career_plays        total college QBR-tracked plays (experience proxy)
    qb_seasons             seasons with QBR data
    qb_final_qbr           Total QBR in the final pre-draft season
    qb_best_qbr            best single-season Total QBR
    qb_epa_per_play        career EPA per play
    qb_final_epa_per_play  final-season EPA per play (recency)
    qb_sack_epa_per_play   career sack EPA per play (pressure-to-sack proxy;
                           more negative = worse)
    qb_run_epa_per_play    career rushing EPA per play (dual-threat value)
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from download_source_data import download_text
from pipeline import ROOT

QBR_URL = "https://raw.githubusercontent.com/JackLich10/nfl-draft-data/main/college_qbr.csv"
PROSPECTS_URL = "https://raw.githubusercontent.com/JackLich10/nfl-draft-data/main/nfl_draft_prospects.csv"
OUT_PATH = ROOT / "data" / "production" / "qb_production.csv"

QB_QBR_FEATURES = [
    "qb_career_plays",
    "qb_seasons",
    "qb_final_qbr",
    "qb_best_qbr",
    "qb_epa_per_play",
    "qb_final_epa_per_play",
    "qb_sack_epa_per_play",
    "qb_run_epa_per_play",
]


def build_qb_production(qbr: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    qbs = prospects[prospects["pos_abbr"].astype(str).str.upper().eq("QB")].copy()
    qbs = qbs[qbs["guid"].notna() & qbs["draft_year"].notna()]
    qbs["draft_year"] = pd.to_numeric(qbs["draft_year"], errors="coerce").astype(int)

    qbr = qbr.copy()
    qbr["season"] = pd.to_numeric(qbr["season"], errors="coerce")
    for col in ["total_qbr", "qb_plays", "total_epa", "run", "sack"]:
        qbr[col] = pd.to_numeric(qbr[col], errors="coerce")

    qbs = qbs.rename(columns={"player_name": "prospect_name", "school": "prospect_school"})
    merged = qbs[["guid", "draft_year", "prospect_name", "pos_abbr", "prospect_school"]].merge(qbr, on="guid", how="inner")
    merged = merged[merged["season"] < merged["draft_year"]]

    rows: list[dict] = []
    for (guid, year), g in merged.groupby(["guid", "draft_year"]):
        g = g.sort_values("season")
        plays = float(g["qb_plays"].sum())
        if plays <= 0:
            continue
        final = g.iloc[-1]
        final_plays = float(final["qb_plays"]) if final["qb_plays"] > 0 else np.nan
        rows.append(
            {
                "Year": int(year),
                "Player": str(g["prospect_name"].iloc[0]).strip(),
                "Pos": "QB",
                "College": str(g["prospect_school"].iloc[0]),
                "qb_career_plays": plays,
                "qb_seasons": int(g["season"].nunique()),
                "qb_final_qbr": float(final["total_qbr"]),
                "qb_best_qbr": float(g["total_qbr"].max()),
                "qb_epa_per_play": float(g["total_epa"].sum()) / plays,
                "qb_final_epa_per_play": (float(final["total_epa"]) / final_plays) if np.isfinite(final_plays) else np.nan,
                "qb_sack_epa_per_play": float(g["sack"].sum()) / plays,
                "qb_run_epa_per_play": float(g["run"].sum()) / plays,
            }
        )
    out = pd.DataFrame(rows).sort_values(["Year", "Player"])
    return out.drop_duplicates(["Year", "Player"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    print(f"Downloading {QBR_URL}")
    qbr = pd.read_csv(io.StringIO(download_text(QBR_URL)))
    print(f"Downloading {PROSPECTS_URL}")
    prospects = pd.read_csv(io.StringIO(download_text(PROSPECTS_URL)), low_memory=False)

    out = build_qb_production(qbr, prospects)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(out_path, index=False)

    cov = out.groupby("Year").size()
    print(f"Wrote {out_path} rows={len(out)}")
    print(json.dumps({str(k): int(v) for k, v in cov.items()}, indent=2))


if __name__ == "__main__":
    main()
