"""Download college football data from collegefootballdata.com (CFBD).

Requires a free API key (https://collegefootballdata.com/key) in the
CFBD_API_KEY environment variable. Locally: `export CFBD_API_KEY=...`;
in GitHub Actions: add it as a repository secret named CFBD_API_KEY.

Pulls per season into data/cfbd/ (gitignored):
    player_season_stats_<year>.csv   full player season stats (all teams)
    team_talent_<year>.csv           team talent composite
    recruiting_<year>.csv            player recruiting ranks/stars

These unlock team-share (dominator), trajectory, level-of-competition, and
recruiting-pedigree features with history back to ~2004 - the inputs the
free GitHub mirrors only carry from 2014.

Usage:
    python src/download_cfbd.py --start-year 2004 --end-year 2025
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

from pipeline import ROOT

BASE = "https://api.collegefootballdata.com"
OUT_DIR = ROOT / "data" / "cfbd"

ENDPOINTS = {
    "player_season_stats": "/stats/player/season?year={year}",
    "team_talent": "/talent?year={year}",
    "recruiting": "/recruiting/players?year={year}",
}


def fetch_json(path: str, api_key: str, retries: int = 3) -> list:
    url = BASE + path
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "APEX-NFLModel/1.0",
            })
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"CFBD request failed: {url}") from last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--endpoints", type=str, default=",".join(ENDPOINTS))
    args = parser.parse_args()

    api_key = os.environ.get("CFBD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "CFBD_API_KEY is not set. Get a free key at "
            "https://collegefootballdata.com/key and export it, or add it as a "
            "GitHub Actions secret named CFBD_API_KEY."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [e.strip() for e in args.endpoints.split(",") if e.strip() in ENDPOINTS]
    for year in range(args.start_year, args.end_year + 1):
        for name in wanted:
            out_path = OUT_DIR / f"{name}_{year}.csv"
            if out_path.exists():
                continue
            data = fetch_json(ENDPOINTS[name].format(year=year), api_key)
            frame = pd.json_normalize(data)
            frame.to_csv(out_path, index=False)
            print(f"Wrote {out_path} rows={len(frame):,}")
            time.sleep(0.5)  # stay well under rate limits

    print("Done. Feature builders can now read data/cfbd/*.csv")


if __name__ == "__main__":
    main()
