"""Convert PFF NCAA exports into position production features.

Drop PFF exports (xlsx or csv) into data/pff/. Expected shape: the
`pff_ncaa_all_positions_2015_2025` table (player-season rows with `_report`,
`_season`, `draft_season`, `player`, `position` and PFF metric columns).
Exports whose first row is a "Table name: ..." banner are handled.

For each player/draft class this script aggregates seasons strictly before the
draft year (career weighted by snaps where sensible, plus final season) and
merges the mapped features into data/production/*.csv so the position models
pick them up automatically.

PFF data is licensed: data/pff/ and the feature files carrying PFF values are
gitignored and must never be committed or published raw. Model scores trained
on them are fine to publish.

Currently mapped reports:
    passing_grades  -> QB features (pressure-to-sack, TWP, BTT, accuracy, grade)

Other report types in the full export (receiving, rushing, blocking, defense)
are listed with their columns when encountered so mappings can be added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT

PFF_DIR = ROOT / "data" / "pff"
PRODUCTION_DIR = ROOT / "data" / "production"

QB_PASSING_MAP = {
    "pressure_to_sack_rate": "qb_pressure_to_sack_rate",
    "sack_percent": "qb_sack_rate",
    "btt_rate": "qb_big_time_throw_rate",
    "twp_rate": "qb_turnover_worthy_play_rate",
    "accuracy_percent": "qb_adj_completion_pct",
    "comp_pct_diff": "qb_cpoe",
    "grades_pass": "qb_pff_pass_grade",
}


def read_pff_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, nrows=2)
        header_row = 1 if str(raw.iloc[0, 0]).startswith("Table name") else 0
        return pd.read_excel(path, header=header_row)
    raw = pd.read_csv(path, header=None, nrows=2, low_memory=False)
    header_row = 1 if str(raw.iloc[0, 0]).startswith("Table name") else 0
    return pd.read_csv(path, header=header_row, low_memory=False)


def load_pff() -> pd.DataFrame:
    frames = []
    for path in sorted(PFF_DIR.glob("*")):
        if path.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
            continue
        print(f"Reading {path.name}")
        frames.append(read_pff_file(path))
    if not frames:
        raise FileNotFoundError(f"No PFF exports found in {PFF_DIR}")
    df = pd.concat(frames, ignore_index=True)
    for col in ["_season", "draft_season"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def aggregate_qb_passing(df: pd.DataFrame) -> pd.DataFrame:
    qb = df[df["_report"].astype(str).eq("passing_grades")].copy()
    qb = qb[qb["draft_season"].notna() & qb["player"].notna()]
    qb = qb[qb["_season"] < qb["draft_season"]]
    if qb.empty:
        return pd.DataFrame()
    qb["dropbacks"] = pd.to_numeric(qb.get("dropbacks"), errors="coerce").fillna(0.0)
    for src in QB_PASSING_MAP:
        qb[src] = pd.to_numeric(qb.get(src), errors="coerce")

    rows = []
    for (player, draft_year), g in qb.groupby(["player", "draft_season"]):
        g = g.sort_values("_season")
        final = g.iloc[-1]
        weights = g["dropbacks"].clip(lower=1)
        row = {
            "Year": int(draft_year),
            "Player": str(player).strip(),
            "Pos": "QB",
            "qb_pff_dropbacks": float(g["dropbacks"].sum()),
        }
        for src, dst in QB_PASSING_MAP.items():
            values = g[src]
            mask = values.notna()
            row[dst] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan
            row[f"{dst}_final"] = float(final[src]) if pd.notna(final[src]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def merge_into_production(features: pd.DataFrame, filename: str) -> Path:
    path = PRODUCTION_DIR / filename
    if path.exists():
        existing = pd.read_csv(path)
        merged = existing.merge(
            features.drop(columns=[c for c in ["Pos"] if c in features.columns]),
            on=["Year", "Player"],
            how="outer",
            suffixes=("", "_pff_dup"),
        )
        merged = merged[[c for c in merged.columns if not c.endswith("_pff_dup")]]
    else:
        merged = features
    merged.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    df = load_pff()
    print(f"PFF rows: {len(df):,} | seasons {int(df['_season'].min())}-{int(df['_season'].max())}")
    reports = df["_report"].astype(str).value_counts().to_dict()
    print("Reports found:", reports)

    qb = aggregate_qb_passing(df)
    if not qb.empty:
        path = merge_into_production(qb, "qb_production.csv")
        cov = qb.groupby("Year").size().to_dict()
        print(f"Merged {len(qb)} QB rows into {path}; classes: {cov}")

    unmapped = [r for r in reports if r != "passing_grades"]
    if unmapped:
        print("\nUnmapped report types (add mappings in this script):")
        for report in unmapped:
            cols = df.loc[df["_report"].astype(str).eq(report)].dropna(axis=1, how="all").columns
            print(f"  {report}: {len(cols)} non-empty columns")


if __name__ == "__main__":
    main()
