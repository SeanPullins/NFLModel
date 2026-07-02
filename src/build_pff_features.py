"""Convert PFF NCAA exports into position production features.

Drop PFF exports into data/pff/ (gitignored - licensed data, never commit):

1. Per-season CSVs exported from PFF position pages, named like
   `rushing-grades__HB-FB-QB__2022.csv` (season parsed from the filename).
2. The `pff_ncaa_all_positions_*` xlsx table (has `_season` per row).

Season rows are matched to draft classes through data/draft_data.csv by
normalized player name: a drafted player's PFF seasons are those in the five
years before his draft year. This works even when only some seasons have been
exported. Ambiguous names (two drafted players in the same window) are skipped
and logged.

Currently mapped: QB passing/rushing metrics -> data/production/qb_production.csv
(merged alongside the ESPN QBR features). Other report types are surfaced with
row counts so mappings can be added as fuller exports arrive.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, norm

PFF_DIR = ROOT / "data" / "pff"
PRODUCTION_DIR = ROOT / "data" / "production"
SEASON_FILE_RE = re.compile(r"__(\d{4})\.csv$")
SEASON_WINDOW = 5  # college seasons can precede the draft by up to this many years

QB_METRIC_MAP = {
    "pressure_to_sack_rate": "qb_pressure_to_sack_rate",
    "sack_percent": "qb_sack_rate",
    "btt_rate": "qb_big_time_throw_rate",
    "twp_rate": "qb_turnover_worthy_play_rate",
    "accuracy_percent": "qb_adj_completion_pct",
    "comp_pct_diff": "qb_cpoe",
    "grades_pass": "qb_pff_pass_grade",
    "grades_run": "qb_pff_run_grade",
    "grades_offense": "qb_pff_offense_grade",
}


def read_banner_aware(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, nrows=2)
        header_row = 1 if str(raw.iloc[0, 0]).startswith("Table name") else 0
        return pd.read_excel(path, header=header_row)
    raw = pd.read_csv(path, header=None, nrows=2, low_memory=False)
    header_row = 1 if str(raw.iloc[0, 0]).startswith("Table name") else 0
    return pd.read_csv(path, header=header_row, low_memory=False)


def load_pff_seasons() -> pd.DataFrame:
    """Normalize every export in data/pff/ into one player-season frame."""
    frames = []
    for path in sorted(PFF_DIR.glob("*")):
        if path.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
            continue
        if path.name.startswith("receiving-"):
            continue  # handled by build_receiving_player_seasons
        df = read_banner_aware(path)
        if "_season" in df.columns:
            df["season"] = pd.to_numeric(df["_season"], errors="coerce")
        else:
            match = SEASON_FILE_RE.search(path.name)
            if not match:
                print(f"Skipping {path.name}: no season column and no season in filename")
                continue
            df["season"] = int(match.group(1))
        df["source_file"] = path.name
        frames.append(df)
        print(f"Read {path.name}: {len(df):,} rows, season(s) {int(df['season'].min())}-{int(df['season'].max())}")
    if not frames:
        raise FileNotFoundError(f"No PFF exports found in {PFF_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["player"].notna() & out["season"].notna()].copy()
    out["season"] = out["season"].astype(int)
    out["name_key"] = out["player"].map(norm)
    # The same player-season can appear in several exports; keep the row with
    # the most populated metric fields.
    metric_cols = [c for c in QB_METRIC_MAP if c in out.columns]
    out["_filled"] = out[metric_cols].notna().sum(axis=1) if metric_cols else 0
    out = out.sort_values("_filled", ascending=False).drop_duplicates(["name_key", "season"], keep="first")
    return out.drop(columns=["_filled"])


def load_draft_classes() -> pd.DataFrame:
    draft = pd.read_csv(ROOT / "data" / "draft_data.csv")
    draft = draft[draft["Player"].notna() & draft["Year"].notna()].copy()
    draft["Year"] = pd.to_numeric(draft["Year"], errors="coerce").astype(int)
    draft["name_key"] = draft["Player"].map(norm)
    return draft


def match_to_draft_classes(pff: pd.DataFrame, draft: pd.DataFrame, positions: tuple[str, ...]) -> pd.DataFrame:
    """Attach each PFF season row to the drafted player it belongs to."""
    scope = draft[draft["Pos"].astype(str).str.upper().isin(positions)]
    counts = scope.groupby("name_key")["Year"].nunique()
    ambiguous = set(counts[counts > 1].index)

    merged = pff.merge(
        scope[["name_key", "Year", "Player", "Pos"]],
        on="name_key",
        how="inner",
        suffixes=("", "_draft"),
    )
    window = (merged["season"] < merged["Year"]) & (merged["season"] >= merged["Year"] - SEASON_WINDOW)
    merged = merged[window]
    ambiguous_used = merged[merged["name_key"].isin(ambiguous)]
    if len(ambiguous_used):
        multi = merged[merged["name_key"].isin(ambiguous)].groupby("name_key")["Year"].nunique()
        drop_keys = set(multi[multi > 1].index)
        if drop_keys:
            print(f"Skipping {len(drop_keys)} ambiguous names matching multiple draft classes: {sorted(drop_keys)[:5]}...")
            merged = merged[~merged["name_key"].isin(drop_keys)]
    return merged


def aggregate_qb(matched: pd.DataFrame) -> pd.DataFrame:
    qb = matched[matched["position"].astype(str).str.upper().eq("QB")].copy()
    if qb.empty:
        return pd.DataFrame()
    qb["dropbacks"] = pd.to_numeric(qb.get("dropbacks"), errors="coerce").fillna(0.0)
    for src in QB_METRIC_MAP:
        if src in qb.columns:
            qb[src] = pd.to_numeric(qb[src], errors="coerce")

    rows = []
    for (name_key, year), g in qb.groupby(["name_key", "Year"]):
        g = g.sort_values("season")
        final = g.iloc[-1]
        weights = g["dropbacks"].clip(lower=1)
        row = {
            "Year": int(year),
            "Player": str(final["Player"]).strip(),
            "Pos": "QB",
            "qb_pff_dropbacks": float(g["dropbacks"].sum()),
            "qb_pff_seasons": int(g["season"].nunique()),
        }
        for src, dst in QB_METRIC_MAP.items():
            if src not in g.columns:
                continue
            values = g[src]
            mask = values.notna()
            row[dst] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan
            row[f"{dst}_final"] = float(final[src]) if pd.notna(final[src]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


RECEIVING_FILE_RE = re.compile(r"receiving-(concept|depth|scheme)__.*__(\d{4})\.csv$")
RECEIVING_POS_GROUP = {"WR": "wr", "TE": "te", "HB": "rb", "FB": "rb"}


def _split_sum(df: pd.DataFrame, prefixes: tuple[str, ...], stat: str) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for prefix in prefixes:
        col = f"{prefix}_{stat}"
        if col in df.columns:
            total = total + pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return total


def _split_weighted(df: pd.DataFrame, prefixes: tuple[str, ...], stat: str, weight_stat: str = "routes") -> pd.Series:
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for prefix in prefixes:
        vcol, wcol = f"{prefix}_{stat}", f"{prefix}_{weight_stat}"
        if vcol not in df.columns or wcol not in df.columns:
            continue
        v = pd.to_numeric(df[vcol], errors="coerce")
        w = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0)
        mask = v.notna()
        num[mask] += (v * w)[mask]
        den[mask] += w[mask]
    return (num / den.replace(0, np.nan))


def build_receiving_player_seasons() -> pd.DataFrame:
    """One row per pass-catcher per season with overall metrics rebuilt from
    the concept (slot/screen/wide), scheme (man/zone), and depth splits."""
    tables: dict[tuple[str, int], pd.DataFrame] = {}
    for path in sorted(PFF_DIR.glob("receiving-*.csv")):
        match = RECEIVING_FILE_RE.search(path.name)
        if not match:
            continue
        tables[(match.group(1), int(match.group(2)))] = pd.read_csv(path, low_memory=False)

    rows = []
    seasons = sorted({season for (_, season) in tables})
    align = ("slot", "screen", "wide")
    for season in seasons:
        concept = tables.get(("concept", season))
        if concept is None:
            continue
        c = concept.copy()
        c["routes"] = _split_sum(c, align, "routes")
        c["targets"] = _split_sum(c, align, "targets")
        c["yards"] = _split_sum(c, align, "yards")
        c["receptions"] = _split_sum(c, align, "receptions")
        c["drops"] = _split_sum(c, align, "drops")
        c["contested_rec"] = _split_sum(c, ("slot", "wide"), "contested_receptions")
        c["contested_tgt"] = _split_sum(c, ("slot", "wide"), "contested_targets")
        c["yac"] = _split_sum(c, align, "yards_after_catch")
        c["route_grade"] = _split_weighted(c, align, "grades_pass_route")
        c["yprr"] = c["yards"] / c["routes"].replace(0, np.nan)
        c["slot_route_share"] = pd.to_numeric(c.get("slot_routes"), errors="coerce").fillna(0.0) / c["routes"].replace(0, np.nan)
        c["drop_rate"] = c["drops"] / (c["receptions"] + c["drops"]).replace(0, np.nan)
        c["contested_catch_rate"] = c["contested_rec"] / c["contested_tgt"].replace(0, np.nan)
        c["yac_per_rec"] = c["yac"] / c["receptions"].replace(0, np.nan)

        keep = c[["player", "player_id", "position", "team_name", "routes", "targets", "yards",
                  "yprr", "route_grade", "slot_route_share", "drop_rate", "contested_catch_rate",
                  "yac_per_rec"]].copy()

        scheme = tables.get(("scheme", season))
        if scheme is not None:
            s = scheme.copy()
            s["man_yprr"] = pd.to_numeric(s.get("man_yprr"), errors="coerce")
            keep = keep.merge(s[["player_id", "man_yprr"]], on="player_id", how="left")

        depth = tables.get(("depth", season))
        if depth is not None:
            d = depth.copy()
            d["deep_targets"] = pd.to_numeric(d.get("deep_targets"), errors="coerce")
            d["all_depth_targets"] = _split_sum(d, ("deep", "medium", "short", "behind_los"), "targets")
            d["deep_target_share"] = d["deep_targets"] / d["all_depth_targets"].replace(0, np.nan)
            keep = keep.merge(d[["player_id", "deep_target_share"]], on="player_id", how="left")

        keep["season"] = season
        rows.append(keep)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[out["routes"] >= 50]  # ignore trick-play / special-teams noise
    out["name_key"] = out["player"].map(norm)
    return out.drop_duplicates(["name_key", "season"], keep="first")


RECEIVING_METRICS = ["yprr", "route_grade", "slot_route_share", "drop_rate",
                     "contested_catch_rate", "yac_per_rec", "man_yprr", "deep_target_share"]


def aggregate_receiving(matched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Route-weighted career + final-season receiving features per draft class,
    split into wr/te/rb production tables."""
    out: dict[str, list[dict]] = {"wr": [], "te": [], "rb": []}
    for (name_key, year), g in matched.groupby(["name_key", "Year"]):
        g = g.sort_values("season")
        final = g.iloc[-1]
        group = RECEIVING_POS_GROUP.get(str(final["position"]).upper())
        if group is None:
            continue
        weights = pd.to_numeric(g["routes"], errors="coerce").clip(lower=1)
        row = {
            "Year": int(year),
            "Player": str(final["Player"]).strip(),
            f"{group}_pff_routes": float(g["routes"].sum()),
            f"{group}_pff_seasons": int(g["season"].nunique()),
        }
        for metric in RECEIVING_METRICS:
            if metric not in g.columns:
                continue
            values = pd.to_numeric(g[metric], errors="coerce")
            mask = values.notna()
            name = "yards_per_route_run" if metric == "yprr" else f"pff_{metric}"
            row[f"{group}_{name}"] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else np.nan
            row[f"{group}_{name}_final"] = float(final[metric]) if pd.notna(final[metric]) else np.nan
        out[group].append(row)
    return {k: pd.DataFrame(v) for k, v in out.items() if v}


def merge_into_production(features: pd.DataFrame, filename: str) -> Path:
    path = PRODUCTION_DIR / filename
    if path.exists():
        existing = pd.read_csv(path)
        drop = [c for c in features.columns if c in existing.columns and c not in ("Year", "Player")]
        existing = existing.drop(columns=drop)
        merged = existing.merge(features.drop(columns=["Pos"], errors="ignore"), on=["Year", "Player"], how="outer")
    else:
        merged = features
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    pff = load_pff_seasons()
    draft = load_draft_classes()

    matched = match_to_draft_classes(pff, draft, positions=("QB", "HB", "FB", "RB", "WR", "TE"))
    qb = aggregate_qb(matched)
    if not qb.empty:
        path = merge_into_production(qb, "qb_production.csv")
        print(f"Merged {len(qb)} QB draft-class rows into {path}")
        print("QB classes covered:", qb.groupby("Year").size().to_dict())

    receiving = build_receiving_player_seasons()
    if not receiving.empty:
        rec_matched = match_to_draft_classes(receiving, draft, positions=("WR", "TE", "HB", "FB", "RB"))
        rec_features = aggregate_receiving(rec_matched)
        for group, frame in rec_features.items():
            path = merge_into_production(frame, f"{group}_production.csv")
            print(f"Merged {len(frame)} {group.upper()} draft-class rows into {path}: classes {frame.groupby('Year').size().to_dict()}")
        rec_seasons = sorted(receiving["season"].unique())
        print("Receiving seasons on hand:", rec_seasons)

    seasons = sorted(pff["season"].unique())
    print("\nPFF passing seasons on hand:", seasons)
    missing = [y for y in range(2014, 2026) if y not in seasons]
    if missing:
        print("Missing seasons for full QB coverage:", missing)


if __name__ == "__main__":
    main()
