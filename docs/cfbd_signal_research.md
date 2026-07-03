# CFBD college-production signal research

Purpose: use the repo's CollegeFootballData access to test whether college production signals help forecast NFL outcomes for recent draft classes without silently changing the public APEX score.

## Agent assignments

| Agent | Job | Output |
|---|---|---|
| CFBD Data Agent | Download `/stats/player/season` by year using the existing repo secret/environment key. | `data/production/cfbd_player_seasons.csv` |
| Feature Engineering Agent | Convert player-season stats into pre-draft prospect features using only seasons before draft year. | `data/production/cfbd_production.csv` |
| Signal Audit Agent | Test each feature against career outcome after controlling for draft slot. | `reports/cfbd_signal_audit.csv` |
| Era Stability Agent | Check whether signs hold across 2004-2009, 2010-2015, and 2016-2021. | `era_values`, `era_stability` |
| Recent Pick Agent | Apply historically interesting signals to newer classes without treating immature NFL outcomes as proof. | `reports/cfbd_recent_pick_notes.csv` |

## How to run

```bash
python src/build_cfbd_production.py --start-year 2004 --end-year 2026
python src/build_features.py --end-year 2021
python src/cfbd_signal_audit.py
```

The builder looks for one of these environment variables:

```text
CFBD_API_KEY
COLLEGE_FOOTBALL_DATA_API_KEY
CFB_DATA_API_KEY
```

The key is never printed. If no key is present, the script writes a skipped report instead of failing the whole free public build.

## What we test first

The first pass focuses on stats that should travel across time better than raw totals:

- efficiency: yards per attempt, yards per carry, yards per reception, yards per touch
- usage/durability: seasons, touches, final-year volume
- scoring efficiency: touchdowns per touch, passing TD rate, interception rate
- defensive playmaking: sacks + tackles for loss + interceptions + pass breakups + forced fumbles
- final-season vs career forms to test whether recency helps

## Promotion contract

No CFBD signal becomes a public APEX score until it passes the same promotion gates:

| Gate | Requirement |
|---|---|
| Residual value | Positive after controlling for draft slot/log pick |
| Era stability | Same directional signal across multiple historical eras |
| Sample size | Enough drafted players to avoid one-class mirages |
| Rolling validation | Future model backtest beats the pick-only market with acceptable worst-year drawdown |

Until then, CFBD output is research: candidates, notes, and recent-pick flags only.
