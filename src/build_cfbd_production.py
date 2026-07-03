"""Build optional CollegeFootballData (CFBD) production features.

This script is intentionally research-first:
- it uses only seasons before the player's draft year;
- it never stores or prints the API key;
- it writes optional features to data/production/cfbd_production.csv;
- missing API credentials fail soft by default so public builds remain free.

Expected secret/env var names:
    CFBD_API_KEY
    COLLEGE_FOOTBALL_DATA_API_KEY
    CFB_DATA_API_KEY
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import ROOT, POSMAP, norm

BASE_URL = "https://api.collegefootballdata.com"
DEFAULT_OUT = ROOT / "data" / "production" / "cfbd_production.csv"
RAW_OUT = ROOT / "data" / "production" / "cfbd_player_seasons.csv"
META_OUT = ROOT / "reports" / "cfbd_download_report.json"

API_KEY_ENV_NAMES = ("CFBD_API_KEY", "COLLEGE_FOOTBALL_DATA_API_KEY", "CFB_DATA_API_KEY")

# CFBD stat names are not guaranteed to be identical forever, so normalize
# common category/statType pairs into stable internal names.
STAT_ALIASES = {
    ("passing", "yds"): "pass_yds",
    ("passing", "yards"): "pass_yds",
    ("passing", "td"): "pass_td",
    ("passing", "tds"): "pass_td",
    ("passing", "int"): "pass_int",
    ("passing", "ints"): "pass_int",
    ("passing", "att"): "pass_att",
    ("passing", "attempts"): "pass_att",
    ("passing", "cmp"): "pass_cmp",
    ("passing", "completions"): "pass_cmp",
    ("rushing", "yds"): "rush_yds",
    ("rushing", "yards"): "rush_yds",
    ("rushing", "td"): "rush_td",
    ("rushing", "tds"): "rush_td",
    ("rushing", "att"): "rush_att",
    ("rushing", "attempts"): "rush_att",
    ("receiving", "yds"): "rec_yds",
    ("receiving", "yards"): "rec_yds",
    ("receiving", "td"): "rec_td",
    ("receiving", "tds"): "rec_td",
    ("receiving", "rec"): "rec_rec",
    ("receiving", "receptions"): "rec_rec",
    ("defensive", "total"): "def_tackles",
    ("defensive", "tot"): "def_tackles",
    ("defensive", "solo"): "def_solo",
    ("defensive", "sack"): "def_sacks",
    ("defensive", "sacks"): "def_sacks",
    ("defensive", "tfl"): "def_tfl",
    ("defensive", "int"): "def_int",
    ("defensive", "ints"): "def_int",
    ("defensive", "pd"): "def_pd",
    ("defensive", "pbu"): "def_pd",
    ("defensive", "ff"): "def_ff",
}

FEATURE_COLUMNS = [
    "cfbd_seasons",
    "cfbd_best_total_yards",
    "cfbd_final_total_yards",
    "cfbd_total_td",
    "cfbd_final_total_td",
    "cfbd_touch_volume",
    "cfbd_scrimmage_yards_per_touch",
    "cfbd_final_scrimmage_yards_per_touch",
    "cfbd_td_per_touch",
    "cfbd_pass_ypa",
    "cfbd_final_pass_ypa",
    "cfbd_pass_td_rate",
    "cfbd_pass_int_rate",
    "cfbd_rush_ypc",
    "cfbd_final_rush_ypc",
    "cfbd_rec_ypr",
    "cfbd_final_rec_ypr",
    "cfbd_def_playmaking",
    "cfbd_final_def_playmaking",
    "cfbd_sack_tfl_per_tackle",
]


def api_key_from_env() -> str | None:
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def fetch_json(path: str, params: dict[str, Any], api_key: str, retries: int = 3) -> Any:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{BASE_URL}{path}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "User-Agent": "APEX-NFLModel-CFBD/1.0",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on remote API
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"CFBD request failed for {path} {params}") from last_exc


def clean_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def college_match_score(draft_college: object, cfbd_team: object) -> int:
    a = clean_token(draft_college)
    b = clean_token(cfbd_team)
    if not a or not b:
        return 0
    if a == b:
        return 3
    if a in b or b in a:
        return 2
    a2 = a.replace("state", "st")
    b2 = b.replace("state", "st")
    if a2 == b2 or a2 in b2 or b2 in a2:
        return 1
    return 0


def stat_key(row: dict[str, Any]) -> str | None:
    cat = str(row.get("category") or row.get("statCategory") or "").strip().lower()
    typ = str(row.get("statType") or row.get("stat_type") or row.get("type") or "").strip().lower()
    cat = re.sub(r"[^a-z]", "", cat)
    typ = re.sub(r"[^a-z]", "", typ)
    return STAT_ALIASES.get((cat, typ))


def player_season_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        key = stat_key(item)
        if not key:
            continue
        stat = pd.to_numeric(pd.Series([item.get("stat")]), errors="coerce").iloc[0]
        if pd.isna(stat):
            continue
        rows.append(
            {
                "season": item.get("season") or item.get("year"),
                "player": item.get("player") or item.get("name"),
                "player_id": item.get("playerId") or item.get("id"),
                "team": item.get("team"),
                "conference": item.get("conference"),
                "stat_key": key,
                "stat": float(stat),
            }
        )
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    long["season"] = pd.to_numeric(long["season"], errors="coerce").astype("Int64")
    wide = (
        long.pivot_table(
            index=["season", "player", "player_id", "team", "conference"],
            columns="stat_key",
            values="stat",
            aggfunc="sum",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in set(STAT_ALIASES.values()):
        if col not in wide.columns:
            wide[col] = 0.0
        wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)
    wide["player_key"] = wide["player"].map(norm)
    return add_player_season_features(wide)


def safe_div(num: Any, den: Any) -> float:
    try:
        n = float(num)
        d = float(den)
    except Exception:
        return np.nan
    return n / d if d > 0 else np.nan


def add_player_season_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_yards"] = out["pass_yds"] + out["rush_yds"] + out["rec_yds"]
    out["total_td"] = out["pass_td"] + out["rush_td"] + out["rec_td"]
    out["touches"] = out["rush_att"] + out["rec_rec"]
    out["scrimmage_yards"] = out["rush_yds"] + out["rec_yds"]
    out["offensive_touches"] = out["pass_att"] + out["rush_att"] + out["rec_rec"]
    out["def_playmaking"] = out["def_sacks"] + out["def_tfl"] + out["def_int"] + out["def_pd"] + out["def_ff"]

    out["pass_ypa"] = out.apply(lambda r: safe_div(r.pass_yds, r.pass_att), axis=1)
    out["pass_td_rate"] = out.apply(lambda r: safe_div(r.pass_td, r.pass_att), axis=1)
    out["pass_int_rate"] = out.apply(lambda r: safe_div(r.pass_int, r.pass_att), axis=1)
    out["rush_ypc"] = out.apply(lambda r: safe_div(r.rush_yds, r.rush_att), axis=1)
    out["rec_ypr"] = out.apply(lambda r: safe_div(r.rec_yds, r.rec_rec), axis=1)
    out["scrimmage_yards_per_touch"] = out.apply(lambda r: safe_div(r.scrimmage_yards, r.touches), axis=1)
    out["td_per_touch"] = out.apply(lambda r: safe_div(r.rush_td + r.rec_td, r.touches), axis=1)
    out["sack_tfl_per_tackle"] = out.apply(lambda r: safe_div(r.def_sacks + r.def_tfl, r.def_tackles), axis=1)
    return out


def download_player_seasons(start_year: int, end_year: int, api_key: str, season_type: str = "regular") -> pd.DataFrame:
    frames = []
    status = {}
    for year in range(start_year, end_year + 1):
        records = fetch_json("/stats/player/season", {"year": year, "seasonType": season_type}, api_key)
        frame = player_season_frame(records if isinstance(records, list) else [])
        status[str(year)] = int(len(frame))
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.attrs["download_status"] = status
    return out


def best_candidate_group(g: pd.DataFrame, draft_college: object) -> pd.DataFrame:
    if g.empty:
        return g
    g = g.copy()
    g["college_match"] = g["team"].map(lambda team: college_match_score(draft_college, team))
    if g["college_match"].max() > 0:
        return g[g["college_match"].eq(g["college_match"].max())].copy()
    team_rank = g.groupby("team")["season"].agg(["count", "max"]).sort_values(["count", "max"], ascending=False)
    return g[g["team"].eq(team_rank.index[0])].copy()


def aggregate_for_draft(draft: pd.DataFrame, seasons: pd.DataFrame, lookback: int = 6) -> pd.DataFrame:
    if seasons.empty:
        return pd.DataFrame(columns=["Year", "Player", "College", *FEATURE_COLUMNS])

    draft = draft.copy()
    draft["Year"] = pd.to_numeric(draft["Year"], errors="coerce")
    draft = draft[draft["Year"].notna() & draft["Player"].notna()].copy()
    draft["Year"] = draft["Year"].astype(int)
    draft["player_key"] = draft["Player"].map(norm)
    draft["pos_g"] = draft.get("Pos", "").map(POSMAP).fillna("OTH") if "Pos" in draft.columns else "OTH"

    rows: list[dict[str, Any]] = []
    by_name = {k: g.copy() for k, g in seasons.groupby("player_key")}

    for d in draft.itertuples(index=False):
        player_key = getattr(d, "player_key")
        year = int(getattr(d, "Year"))
        player = getattr(d, "Player")
        college = getattr(d, "College", "")
        g = by_name.get(player_key, pd.DataFrame())
        if g.empty:
            continue
        g = g[(pd.to_numeric(g["season"], errors="coerce") < year) & (pd.to_numeric(g["season"], errors="coerce") >= year - lookback)].copy()
        if g.empty:
            continue
        g = best_candidate_group(g, college).sort_values("season")
        final = g.iloc[-1]
        career = g.fillna(0.0)
        pass_att = career["pass_att"].sum()
        rush_att = career["rush_att"].sum()
        rec_rec = career["rec_rec"].sum()
        touches = career["touches"].sum()
        def_tackles = career["def_tackles"].sum()
        row = {
            "Year": year,
            "Player": str(player).strip(),
            "College": str(college),
            "Pos": getattr(d, "Pos", ""),
            "pos_g": getattr(d, "pos_g", ""),
            "cfbd_seasons": int(career["season"].nunique()),
            "cfbd_best_total_yards": float(career["total_yards"].max()),
            "cfbd_final_total_yards": float(final.get("total_yards", np.nan)),
            "cfbd_total_td": float(career["total_td"].sum()),
            "cfbd_final_total_td": float(final.get("total_td", np.nan)),
            "cfbd_touch_volume": float(touches),
            "cfbd_scrimmage_yards_per_touch": safe_div(career["scrimmage_yards"].sum(), touches),
            "cfbd_final_scrimmage_yards_per_touch": float(final.get("scrimmage_yards_per_touch", np.nan)),
            "cfbd_td_per_touch": safe_div(career["rush_td"].sum() + career["rec_td"].sum(), touches),
            "cfbd_pass_ypa": safe_div(career["pass_yds"].sum(), pass_att),
            "cfbd_final_pass_ypa": float(final.get("pass_ypa", np.nan)),
            "cfbd_pass_td_rate": safe_div(career["pass_td"].sum(), pass_att),
            "cfbd_pass_int_rate": safe_div(career["pass_int"].sum(), pass_att),
            "cfbd_rush_ypc": safe_div(career["rush_yds"].sum(), rush_att),
            "cfbd_final_rush_ypc": float(final.get("rush_ypc", np.nan)),
            "cfbd_rec_ypr": safe_div(career["rec_yds"].sum(), rec_rec),
            "cfbd_final_rec_ypr": float(final.get("rec_ypr", np.nan)),
            "cfbd_def_playmaking": safe_div(career["def_playmaking"].sum(), len(career)),
            "cfbd_final_def_playmaking": float(final.get("def_playmaking", np.nan)),
            "cfbd_sack_tfl_per_tackle": safe_div(career["def_sacks"].sum() + career["def_tfl"].sum(), def_tackles),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Year", "Player", "College", *FEATURE_COLUMNS])
    for col in FEATURE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(["Year", "Player"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--draft-data", default=str(ROOT / "data" / "draft_data.csv"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--raw-out", default=str(RAW_OUT))
    parser.add_argument("--report", default=str(META_OUT))
    parser.add_argument("--lookback", type=int, default=6)
    parser.add_argument("--season-type", default="regular")
    parser.add_argument("--fail-on-missing-key", action="store_true")
    args = parser.parse_args()

    report = {
        "source": "CollegeFootballData /stats/player/season",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "season_type": args.season_type,
        "uses_paid_data_or_apis": False,
        "api_key_printed": False,
    }
    key = api_key_from_env()
    if not key:
        report["status"] = "skipped_missing_api_key"
        report["message"] = f"Set one of {API_KEY_ENV_NAMES} to build CFBD features."
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if args.fail_on_missing_key:
            sys.exit(2)
        return

    draft = pd.read_csv(args.draft_data)
    seasons = download_player_seasons(args.start_year, args.end_year, key, season_type=args.season_type)
    Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
    seasons.round(6).to_csv(args.raw_out, index=False, quoting=csv.QUOTE_MINIMAL)
    out = aggregate_for_draft(draft, seasons, lookback=args.lookback)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(args.out, index=False)
    report.update(
        {
            "status": "ok",
            "raw_player_season_rows": int(len(seasons)),
            "feature_rows": int(len(out)),
            "feature_columns": FEATURE_COLUMNS,
            "download_status": seasons.attrs.get("download_status", {}),
        }
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
