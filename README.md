# APEX Draft Model

Two-stage residual NFL draft model + interactive scouting dashboard for historical NFL draft classes and forward-looking prospect watchlists.

## Current honest validation status

The regenerated public-source backtest showed that **raw APEX has a small edge**, but the old **APEX+ 3.5× residual amplification did not reproduce the earlier headline result**. Until a residual factor passes promotion gates, the honest headline model is **raw APEX**, not APEX+.

Current rule:

> APEX+ is experimental. Raw APEX remains the baseline headline unless `src/sweep_apex_factor.py` finds a factor that passes mean, median, win-rate, and worst-window gates.

## What changed in v1.4 validation fix

- **Removed the hidden year cap** by adding `--end-year` to the backtest scripts.
- **Rolling validation can now truly test 2011-2021** instead of silently stopping at 2016.
- **Added APEX+ factor sweep** with `src/sweep_apex_factor.py`.
- **Promotion is gated**: a factor is promoted only if it passes validation gates.
- **Dashboard/docs are being downgraded away from the old 0.697 APEX+ headline** until regenerated results justify it.

## Existing v1.3 scaffold

- **Optional production feature ingestion** via `src/build_features.py`.
- **Feature registry** for QB/WR/RB/TE/OL/EDGE/DT/LB/DB production columns and consensus-market columns.
- **Pre-draft backtest** using expected pick / consensus rank instead of actual draft slot.
- **Position-specific residual backtest** for QB, skill, OL, front, LB, and DB families.
- **Promotion gates** so a candidate model must improve mean, median, win rate, and worst-window behavior before being promoted.
- **Public source downloader** via `src/download_source_data.py`, which builds the raw files expected by the pipeline from public GitHub data sources.
- **Manual GitHub Action runner** via `.github/workflows/run-backtests.yml`.
- **Feature documentation** in `docs/FEATURES.md`.

## Architecture

1. **Market baseline** — isotonic regression from pick → outcome, with optional per-position blending.
2. **Raw APEX residual model** — 5-seed bagged LightGBM on position-normalized combine/profile features, age, and shrunken college encoding.
3. **Per-position shrinkage** — residual weight tuned by position on an earlier validation fold, then applied to the out-of-time test fold.
4. **APEX+ experimental residual amplification** — `market + factor × (raw APEX - market)`, clipped to a 1-99 percentile range. This is not promoted unless the factor sweep passes gates.

**Target:** within-class Career AV percentile. This is a ranking target, not a calibrated projection of exact career value.

## Repo layout

```text
src/
  pipeline.py                  shared loading, feature, baseline, residual, and metric utilities
  download_source_data.py      downloads public source data and writes pipeline raw inputs
  feature_registry.py          optional production/consensus feature definitions
  build_features.py            builds data/model_features.csv from optional feature files
  improve.py                   train + original holdout evaluation
  backtest.py                  rolling out-of-time validation with --end-year support
  sweep_apex_factor.py         sweeps APEX+ residual factors and gates promotion
  experiment_feature_sets.py   compares profile-only vs postdraft-interaction residual features
  predraft_backtest.py         evaluates true pre-draft market/prospect forecasting
  position_models.py           tests position-family residual models with --end-year support
  validation_gates.py          promotion checks for candidate models
  build_site.py                static dashboard builder
  template.html                dashboard template

.github/workflows/
  run-backtests.yml            downloads sources, runs backtests, factor sweep, gates, uploads reports

data/
  apex_board.csv      generated board used by dashboard
  draft_data.csv      generated raw draft/outcome file after download_source_data.py
  combine_data_pfr_with_stats.csv generated combine/profile file after download_source_data.py
  model_features.csv  generated enriched table after build_features.py
  production/         optional production feature CSVs
  consensus/          optional consensus-board / expected-pick CSVs
  SOURCES.md          raw-data source notes

reports/
  generated holdout, rolling backtest, factor sweep, feature coverage, and experiment outputs
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

- `phcs971/nfl-draft-dataset` → merged combine, draft, career AV, and NCAA data through 2024.
- `array-carpenter/nfl-draft-data` → combine/pro-day measurements through 2026.

## Run from GitHub Actions

Go to:

```text
GitHub repo → Actions → Run APEX Backtests → Run workflow
```

The workflow runs:

```bash
python src/download_source_data.py
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --apex-plus-factor 3.5
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --apex-plus-factor 3.5
python src/validation_gates.py reports/rolling_backtest_summary.csv --out reports/rolling_validation_gates.json
python src/validation_gates.py reports/position_model_backtest_summary.csv --out reports/position_model_validation_gates.json
```

It uploads an artifact named:

```text
apex-backtest-reports
```

## Local validation commands

```bash
pip install -r requirements.txt
python src/download_source_data.py
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --apex-plus-factor 3.5
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --apex-plus-factor 3.5
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
reports/rolling_validation_gates.json
reports/position_model_validation_gates.json
```

## Promotion rule

Promote a candidate only if it improves:

- average lift
- median lift
- window win rate
- worst-window behavior
- practical draft metrics such as precision@32 / precision@64

A single higher headline Spearman is not enough. If no APEX+ factor passes gates, the public claim stays with **raw APEX**.

## Next data additions

Highest-impact next data additions:

1. consensus-board / expected-pick history for true pre-draft forecasting
2. QB pressure-to-sack and age-adjusted efficiency features
3. WR YPRR / target share / breakout age
4. EDGE pressure rate and pass-rush win rate
5. OL pressure allowed and snap data
6. DB coverage and missed-tackle metrics
