# APEX Draft Model

Two-stage residual NFL draft model + interactive scouting dashboard for historical NFL draft classes and forward-looking prospect watchlists.

## Current honest validation status

The regenerated public-source validation found a small, repeatable edge over draft slot, but not enough to justify the older aggressive APEX+ headline. The public/default model is now:

```text
raw APEX + profile feature set
```

That means:

```text
draft market baseline + combine/profile features + age + college encoding
```

NCAA production and APEX+ amplification remain experimental until they beat the default model on average lift, median lift, win rate, and worst-window behavior.

## Current model roles

| Component | Status | Use |
|---|---|---|
| `profile` raw APEX | **Default / public board** | Main board score and headline validation |
| `position_profile_only` | **Top challenger** | Worth tracking, not fully promoted |
| NCAA production features | Experimental | Available for ablations, not headline |
| APEX+ residual amplification | Experimental | Not promoted |

## What changed in this upgrade

- Added `src/feature_sets.py` for explicit model variants.
- Made `profile` the default feature set in `src/backtest.py`.
- Made `profile` the default board feature set in `src/improve.py`.
- Made the position-specific profile-only model the default challenger in `src/position_models.py`.
- Changed validation gates to evaluate raw APEX lift by default.
- Workflow now builds a profile-only public board through 2026 while validating mature classes through 2021.
- NCAA production stays available in ablation reports but is not in the headline board.

## Feature sets

```text
profile                         combine/profile + age + college encoding
profile_plus_production         profile + all NCAA production features
production_only                 all NCAA production features only
offensive_production_only       NCAA offensive production only
defensive_production_only       NCAA defensive production only
```

## Production features available for experiments

The public source includes NCAA career stats. The downloader converts those into pre-draft per-game production features:

```text
college_games
college_pass_yds_pg
college_pass_td_pg
college_pass_int_pg
college_pass_cmp_pct
college_pass_td_int_ratio
college_rush_yds_pg
college_rush_td_pg
college_rec_yds_pg
college_rec_td_pg
college_tackles_pg
college_sacks_pg
college_ints_pg
college_fumbles_pg
college_offensive_yds_pg
college_total_td_pg
college_def_playmaking_pg
```

These come from college production only. They do **not** use NFL outcomes such as Pro Bowls, All-Pro selections, NFL games played, NFL sacks, or NFL receiving/rushing/passing stats.

## Architecture

1. **Market baseline** — isotonic regression from pick to outcome, with optional per-position blending.
2. **Raw APEX residual model** — 5-seed bagged LightGBM on selected feature set.
3. **Per-position shrinkage** — residual weight tuned by position on an earlier validation fold, then applied to the out-of-time test fold.
4. **APEX+ experimental residual amplification** — `market + factor x (raw APEX - market)`, clipped to a 1-99 percentile range. This is not promoted unless factor sweep passes gates.

**Target:** within-class Career AV percentile. This is a ranking target, not a calibrated projection of exact career value.

## Repo layout

```text
src/
  feature_sets.py              named model feature sets
  pipeline.py                  shared loading, feature, baseline, residual, and metric utilities
  download_source_data.py      downloads public source data and writes pipeline raw inputs
  feature_registry.py          optional production/consensus feature definitions
  build_features.py            builds data/model_features.csv from optional feature files
  improve.py                   trains profile-only public board by default
  backtest.py                  rolling out-of-time validation with --feature-set and --end-year support
  sweep_apex_factor.py         sweeps APEX+ residual factors and gates promotion
  ablation_backtest.py         compares profile/production/position-specific feature families
  predraft_backtest.py         evaluates true pre-draft market/prospect forecasting
  position_models.py           tests position-family residual models with --feature-set support
  validation_gates.py          promotion checks for candidate models
  build_site.py                static dashboard builder
  template.html                dashboard template

.github/workflows/
  run-backtests.yml            downloads sources, builds board, runs backtests, factor sweep, ablations, gates, uploads reports

data/
  apex_board.csv      generated board used by dashboard
  draft_data.csv      generated raw draft/outcome file after download_source_data.py
  combine_data_pfr_with_stats.csv generated combine/profile + NCAA production file after download_source_data.py
  model_features.csv  generated enriched table after build_features.py
  production/         optional production feature CSVs
  consensus/          optional consensus-board / expected-pick CSVs
  SOURCES.md          raw-data source notes

reports/
  generated holdout, rolling backtest, factor sweep, feature ablation, feature coverage, and experiment outputs
```

## Raw data setup

Build raw files automatically from public source repos:

```bash
python src/download_source_data.py
```

This writes:

```text
data/draft_data.csv
data/combine_data_pfr_with_stats.csv
```

Current source inputs:

- `phcs971/nfl-draft-dataset` — combine, draft, career AV, and NCAA production data through 2024.
- `array-carpenter/nfl-draft-data` — combine/pro-day measurements through 2026.

## Run from GitHub Actions

Go to:

```text
GitHub repo -> Actions -> Run APEX Backtests -> Run workflow
```

The workflow runs:

```bash
python src/download_source_data.py
python src/improve.py --feature-set profile --end-year 2026
python src/build_site.py
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile --apex-plus-factor 3.5
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile --apex-plus-factor 3.5
python src/ablation_backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021
python src/validation_gates.py reports/rolling_backtest_summary.csv --delta-col delta_raw_vs_pick_spearman_drafted --out reports/rolling_validation_gates.json
python src/validation_gates.py reports/position_model_backtest_summary.csv --delta-col delta_raw_vs_pick_spearman_drafted --out reports/position_model_validation_gates.json
```

It uploads an artifact named:

```text
apex-backtest-reports
```

## Local validation commands

```bash
pip install -r requirements.txt
python src/download_source_data.py
python src/improve.py --feature-set profile --end-year 2026
python src/build_site.py
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile --apex-plus-factor 3.5
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile --apex-plus-factor 3.5
python src/ablation_backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021
```

Key outputs:

```text
reports/rolling_backtest_summary.csv
reports/rolling_backtest_report.json
reports/apex_factor_sweep_summary.csv
reports/apex_factor_sweep_by_year.csv
reports/apex_factor_sweep_report.json
reports/position_model_backtest_summary.csv
reports/position_model_backtest_report.json
reports/feature_ablation_summary.csv
reports/feature_ablation_by_year.csv
reports/feature_ablation_report.json
reports/rolling_validation_gates.json
reports/position_model_validation_gates.json
data/apex_board.csv
index.html
docs/index.html
```

## Feature ablation checks

`src/ablation_backtest.py` compares:

- `global_profile_only`
- `global_profile_plus_all_production`
- `global_production_only`
- `global_offensive_production_only`
- `global_defensive_production_only`
- `position_profile_only`
- `position_profile_plus_all_production`

The main comparison metric is:

```text
delta_candidate_vs_pick_spearman_drafted
```

A production feature family should not be promoted unless it beats profile-only on average lift, median lift, win rate, and worst-window behavior.

## Promotion rule

Promote a candidate only if it improves:

- average lift
- median lift
- window win rate
- worst-window behavior
- practical draft metrics such as precision@32 / precision@64

A single higher headline Spearman is not enough. If no APEX+ factor passes gates, the public claim stays with **raw profile-only APEX**.

## Next data additions

Highest-impact next data additions:

1. consensus-board / expected-pick history for true pre-draft forecasting
2. pressure-to-sack and age-adjusted QB efficiency features
3. route-level WR/TE features such as YPRR and target share
4. EDGE pressure rate and pass-rush win rate
5. OL pressure allowed and snap data
6. DB coverage and missed-tackle metrics
