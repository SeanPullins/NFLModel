"""Download and normalize public NFL draft source data for APEX.

This creates the two raw files expected by src/pipeline.py:

    data/draft_data.csv
    data/combine_data_pfr_with_stats.csv

Primary source:
    phcs971/nfl-draft-dataset, nfl_data.csv

Optional combine/pro-day overlay:
    array-carpenter/nfl-draft-data, data/combine_pro_day.csv

The phcs971 source includes NCAA career production columns. This script converts
those into pre-draft per-game production features and excludes NFL career stats
from the model input to avoid outcome leakage.
"""
from __future__ import annotations

import argparse
import io
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, COLLEGE_PRODUCTION_FEATURES

PHCS_NFL_DATA_URL = "https://raw.githubusercontent.com/phcs971/nfl-draft-dataset/main/nfl_data.csv"
ARRAY_COMBINE_URL = "https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_pro_day.csv"
NFLVERSE_DRAFT_PICKS_URL = "https://raw.githubusercontent.com/nflverse/nflverse-data/releases/draft_picks.csv"

YEAR_RE = re.compile(r"^\d{4}[;,]")


def download_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "APEX-NFLModel/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def repair_year_rows(text: str, delimiter: str) -> str:
    """Join malformed continuation lines until the next row starts with YYYY+delimiter."""
    lines = text.splitlines()
    if not lines:
        return text
    out = [lines[0]]
    current = ""
    row_re = re.compile(rf"^\d{{4}}{re.escape(delimiter)}")
    for line in lines[1:]:
        if row_re.match(line):
            if current:
                out.append(current)
            current = line
        else:
            current += " " + line.strip()
    if current:
        out.append(current)
    return "\n".join(out) + "\n"


def read_repaired_csv(text: str, delimiter: str) -> pd.DataFrame:
    repaired = repair_year_rows(text, delimiter)
    return pd.read_csv(io.StringIO(repaired), sep=delimiter, low_memory=False)


def clean_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").fillna(0.0)
    den = pd.to_numeric(den, errors="coerce")
    return np.where(den.gt(0), num / den, 0.0)


def normalize_position(pos: object) -> str:
    text = str(pos or "").upper().strip()
    mapping = {
        "HB": "RB", "FB": "RB", "TB": "RB", "RB": "RB",
        "S": "DB", "FS": "DB", "SS": "DB", "SAF": "DB", "CB": "DB", "DB": "DB",
        "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
        "DE": "EDGE", "EDGE": "EDGE", "DL": "EDGE", "NT": "DT", "DT": "DT",
        "C": "OL", "G": "OL", "OG": "OL", "OT": "OL", "T": "OL", "OL": "OL",
        "QB": "QB", "WR": "WR", "TE": "TE",
        "K": "ST", "P": "ST", "LS": "ST",
    }
    return mapping.get(text, text or "OTH")


def add_college_production_features(base: pd.DataFrame) -> pd.DataFrame:
    """Create pre-draft production features from NCAA career stats.

    Uses only college fields from the public source. NFL career columns are never
    used as features because they are the outcome being predicted.
    """
    out = base.copy()
    for col in [
        "college_games",
        "college_pass_yds", "college_pass_td", "college_pass_int", "college_pass_cmp_pct",
        "college_rush_yds", "college_rush_td",
        "college_rec_yds", "college_rec_td",
        "college_tackles", "college_sacks", "college_ints", "college_fumbles",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    games = out["college_games"].replace(0, np.nan)
    out["college_pass_yds_pg"] = safe_div(out["college_pass_yds"], games)
    out["college_pass_td_pg"] = safe_div(out["college_pass_td"], games)
    out["college_pass_int_pg"] = safe_div(out["college_pass_int"], games)
    out["college_pass_cmp_pct"] = pd.to_numeric(out["college_pass_cmp_pct"], errors="coerce").fillna(0.0)
    out["college_pass_td_int_ratio"] = out["college_pass_td"] / (out["college_pass_int"] + 1.0)
    out["college_rush_yds_pg"] = safe_div(out["college_rush_yds"], games)
    out["college_rush_td_pg"] = safe_div(out["college_rush_td"], games)
    out["college_rec_yds_pg"] = safe_div(out["college_rec_yds"], games)
    out["college_rec_td_pg"] = safe_div(out["college_rec_td"], games)
    out["college_tackles_pg"] = safe_div(out["college_tackles"], games)
    out["college_sacks_pg"] = safe_div(out["college_sacks"], games)
    out["college_ints_pg"] = safe_div(out["college_ints"], games)
    out["college_fumbles_pg"] = safe_div(out["college_fumbles"], games)
    out["college_offensive_yds_pg"] = safe_div(
        out["college_pass_yds"] + out["college_rush_yds"] + out["college_rec_yds"], games
    )
    out["college_total_td_pg"] = safe_div(
        out["college_pass_td"] + out["college_rush_td"] + out["college_rec_td"], games
    )
    out["college_def_playmaking_pg"] = safe_div(
        out["college_sacks"] + out["college_ints"] + out["college_fumbles"], games
    )

    for col in COLLEGE_PRODUCTION_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_from_phcs(phcs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = add_college_production_features(phcs.copy())
    base["Year"] = clean_numeric(base["year"]).astype("Int64")
    base["Player"] = base["name"].astype(str).str.strip()
    base["College"] = base.get("college", "Unknown").fillna("Unknown").astype(str)
    base["Pos"] = base.get("pos", "OTH").fillna("OTH").astype(str).str.upper()
    base["Pick"] = clean_numeric(base.get("draft_pick"))
    base["Rnd"] = clean_numeric(base.get("draft_round"))
    base["CarAV"] = clean_numeric(base.get("career_av")).fillna(0.0)
    base["Age"] = clean_numeric(base.get("age"))
    base["Team"] = base.get("team")

    draft = base[["Year", "Player", "College", "Pos", "Pick", "Rnd", "Team", "Age", "CarAV"]].copy()
    draft = draft[draft["Year"].notna() & draft["Player"].notna()]
    draft["Year"] = draft["Year"].astype(int)
    draft = draft.sort_values(["Year", "Pick", "Player"], na_position="last").drop_duplicates(["Year", "Player"], keep="first")

    combine = pd.DataFrame(
        {
            "year": base.loc[draft.index, "Year"],
            "player": base.loc[draft.index, "Player"],
            "college": base.loc[draft.index, "College"],
            "position": base.loc[draft.index, "Pos"],
            "height": clean_numeric(base.loc[draft.index, "height"]),
            "weight": clean_numeric(base.loc[draft.index, "weight"]),
            "dash": clean_numeric(base.loc[draft.index, "40_yard"]),
            "vert_leap": clean_numeric(base.loc[draft.index, "vert_leap"]),
            "bench": clean_numeric(base.loc[draft.index, "bench_press"]),
            "broad": clean_numeric(base.loc[draft.index, "broad_jump"]),
            "cone": clean_numeric(base.loc[draft.index, "3_cone"]),
            "shuttle": clean_numeric(base.loc[draft.index, "shuttle"]),
            **{col: clean_numeric(base.loc[draft.index, col]).fillna(0.0) for col in COLLEGE_PRODUCTION_FEATURES},
        }
    )
    combine = combine[combine["year"].notna() & combine["player"].notna()].copy()
    combine["year"] = combine["year"].astype(int)
    combine = combine.drop_duplicates(["year", "player"], keep="first")
    return draft, combine


def normalize_array_combine(array_df: pd.DataFrame) -> pd.DataFrame:
    cols = array_df.columns
    height_col = "Height (in)" if "Height (in)" in cols else "height"
    weight_col = "Weight (lbs)" if "Weight (lbs)" in cols else "weight"
    out = pd.DataFrame(
        {
            "year": clean_numeric(array_df.get("Year")),
            "player": array_df.get("player", array_df.get("Player")).astype(str).str.strip(),
            "college": array_df.get("College", array_df.get("college", "Unknown")).fillna("Unknown").astype(str),
            "position": array_df.get("POS", array_df.get("POS_GP", "OTH")).fillna("OTH").astype(str),
            "height": clean_numeric(array_df.get(height_col)),
            "weight": clean_numeric(array_df.get(weight_col)),
            "dash": clean_numeric(array_df.get("40 Yard")),
            "vert_leap": clean_numeric(array_df.get("Vert Leap (in)")),
            "bench": clean_numeric(array_df.get("Bench Press")),
            "broad": clean_numeric(array_df.get("Broad Jump (in)")),
            "cone": clean_numeric(array_df.get("3Cone")),
            "shuttle": clean_numeric(array_df.get("Shuttle")),
        }
    )
    out = out[out["year"].notna() & out["player"].notna()].copy()
    out["year"] = out["year"].astype(int)
    return out.drop_duplicates(["year", "player"], keep="first")


def overlay_combine(primary: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    key = ["year", "player"]
    base = primary.set_index(key)
    extra = overlay.set_index(key)
    # Overlay only measurement fields. Keep phcs NCAA production features as the production source.
    for col in ["college", "position", "height", "weight", "dash", "vert_leap", "bench", "broad", "cone", "shuttle"]:
        if col not in base.columns or col not in extra.columns:
            continue
        base[col] = base[col].combine_first(extra[col])
        base.update(extra[[col]])
    missing = extra.loc[~extra.index.isin(base.index)]
    for col in COLLEGE_PRODUCTION_FEATURES:
        if col not in missing.columns:
            missing[col] = 0.0
    combined = pd.concat([base, missing], axis=0).reset_index()
    return combined.drop_duplicates(key, keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--skip-array-combine", action="store_true")
    parser.add_argument("--save-source-copies", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {PHCS_NFL_DATA_URL}")
    phcs_text = download_text(PHCS_NFL_DATA_URL)
    if args.save_source_copies:
        (out_dir / "source_phcs_nfl_data.csv").write_text(phcs_text)
    phcs = read_repaired_csv(phcs_text, ";")
    draft, combine = build_from_phcs(phcs)

    if not args.skip_array_combine:
        try:
            print(f"Downloading {ARRAY_COMBINE_URL}")
            array_text = download_text(ARRAY_COMBINE_URL)
            if args.save_source_copies:
                (out_dir / "source_array_combine_pro_day.csv").write_text(array_text)
            array_df = read_repaired_csv(array_text, ",")
            combine = overlay_combine(combine, normalize_array_combine(array_df))
        except Exception as exc:
            print(f"WARNING: array-carpenter combine overlay failed: {exc}")
            print("Continuing with phcs971 combine measurements only.")

    draft_path = out_dir / "draft_data.csv"
    combine_path = out_dir / "combine_data_pfr_with_stats.csv"
    draft.round(6).to_csv(draft_path, index=False)
    combine.round(6).to_csv(combine_path, index=False)

    print(f"Wrote {draft_path} rows={len(draft):,}")
    print(f"Wrote {combine_path} rows={len(combine):,}")
    print("Draft year range:", int(draft["Year"].min()), int(draft["Year"].max()))
    print("Combine year range:", int(combine["year"].min()), int(combine["year"].max()))
    print("Added NCAA production features:", ", ".join(COLLEGE_PRODUCTION_FEATURES))


if __name__ == "__main__":
    main()
