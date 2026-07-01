# APEX Draft Model

Two-stage residual NFL draft model + interactive scouting dashboard covering historical classes and forward-looking prospect watchlists.

The live v1.2.1 board reports **APEX+ Spearman ρ 0.697** on the 2012-14 official holdout versus **0.615** for the pick-only market baseline. Across the four published rolling validation windows from 2008-2020, APEX+ improves over the market in **4 of 4 windows**, with **average lift +0.083** and **median lift +0.080**.

## What changed in v1.3 scaffold

This version adds the infrastructure needed to improve accuracy honestly:

- **Optional production feature ingestion** via `src/build_features.py`.
- **Feature registry** for QB/WR/RB/TE/OL/EDGE/DT/LB/DB production columns and consensus-market columns.
- **Pre-draft backtest** using expected pick / consensus rank instead of actual draft slot.
- **Position-specific residual backtest** for QB, skill, OL, front, LB, and DB families.
- **Promotion gates** so a candidate model must improve mean, median, win rate, and worst-window behavior before being promoted.
- **Public source downloader** via `src/download_source_data.py`, which builds the raw files expected by the pipeline from public GitHub data sources.
- **Manual GitHub Action runner** via `.github/workflows/run-backtests.yml`.
- **Feature documentation** in `docs/FEATURES.md`.

## Existing v1.2+ improvements

- **Fold-safe college encoding** — college strength is fit on the training fold only, then mapped to validation/test/prospect rows.
- **Fold-safe athletic normalization** — position z-scores are learned from the training fold only instead of the full dataframe.
- **Repo-relative paths** — scripts now read from `data/` or `APEX_DATA_DIR` instead of hardcoded scratch paths.
- **Rolling backtests** — `src/backtest.py` reports raw APEX and APEX+ versus pick-only/market baselines.
- **APEX+ validation artifact** — committed CSV/JSON summaries show average and median lift across rolling windows.
- **Public validation page** — `docs/validation.html` shows the rolling-window scoreboard and links to downloadable CSV/JSON.
- **Experiment harness** — `src/experiment_feature_sets.py` tests whether adding post-draft interaction features improves the residual before promoting the change.

## Live dashboard

Deploy free on GitHub Pages:

```bash
git init && git add . && git commit -m "APEX v1.3"
gh repo create apex-draft-model --public --source=. --push
```

Then: repo **Settings → Pages → Source: main / `/docs`**.

The dashboard reads `data/apex_board.csv`, builds a searchable board, and displays pick-vs-outcome, surplus, and player-level scoring views.

Useful public pages:

- `docs/index.html` — board
- `docs/model.html` — model explanation
- `docs/validation.html` — rolling validation summary
- `docs/FEATURES.md` — production/consensus feature guide
- `docs/rolling_backtest_summary.csv` — downloadable validation CSV
- `docs/rolling_backtest_report.json` — downloadable validation JSON

## Architecture

1. **Market baseline** — isotonic regression from pick → outcome, with optional per-position blending.
2. **Athletic/profile residual** — 5-seed bagged LightGBM on position-normalized combine/profile features, age, and shrunken college encoding.
3. **Per-position shrinkage** — residual weight tuned by position on an earlier validation fold, then applied to the out-of-time test fold.
4. **APEX+ residual amplification** — headline projection uses `market + 3.5 × (raw APEX - market)`, clipped to a 1-99 percentile range.

**Target:** within-class Career AV percentile. This is a ranking target, not a calibrated projection of exact career value.

**Important interpretation:** APEX should be judged by whether it improves the rank ordering of prospects over the draft market. The strongest claim is not one holdout score; it is repeated lift over rolling out-of-time windows.

## Repo layout

```text
src/
  pipeline.py                  shared loading, feature, baseline, residual, and metric utilities
  download_source_data.py      downloads public source data and writes pipeline raw inputs
  feature_registry.py          optional production/consensus feature definitions
  build_features.py            builds data/model_features.csv from optional feature files
  improve.py                   v1.2 train + original 2012-14 holdout evaluation
  backtest.py                  rolling out-of-time validation, including APEX+
  experiment_feature_sets.py   compares profile-only vs postdraft-interaction residual features
  predraft_backtest.py         evaluates true pre-draft market/prospect forecasting
  position_models.py           tests position-family residual models
  validation_gates.py          promotion checks for candidate models
  build_site.py                static dashboard builder
  template.html                dashboard template

.github/workflows/
  run-backtests.yml            manual Action: download sources, run backtests, upload reports

data/
  apex_board.csv      generated board used by dashboard
  draft_data.csv      generated raw draft/outcome file after download_source_data.py
  combine_data_pfr_with_stats.csv generated combine/profile file after download_source_data.py
  model_features.csv  generated enriched table after build_features.py
  production/         optional production feature CSVs
  consensus/          optional consensus-board / expected-pick CSVs
  SOURCES.md          raw-data source notes

docs/
  index.html                     GitHub Pages dashboard
  model.html                     model explanation
  validation.html                rolling validation page
  FEATURES.md                    feature upgrade guide
  rolling_backtest_summary.csv   public validation CSV
  rolling_backtest_report.json   public validation JSON
  MODEL_CARD.md                  model assumptions, limits, validation protocol
  VALIDATION.md                  validation guidance and interpretation

models/
  generated LightGBM/isotonic artifacts after running src/improve.py

reports/
  generated holdout, rolling backtest, feature coverage, and experiment outputs
```

## Raw data setup

The training scripts expect these raw files:

```text
draft_data.csv
combine_data_pfr_with_stats.csv
```

Build them automatically from public source repos:

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

You can also point to your own raw-data folder:

```bash
export APEX_DATA_DIR=/path/to/raw-data
```

Source notes are in `data/SOURCES.md`.

## Run from GitHub Actions

After merging the workflow, use:

```text
GitHub repo → Actions → Run APEX Backtests → Run workflow
```

Defaults:

```text
first_test_year = 2011
last_test_year = 2021
apex_plus_factor = 3.5
skip_array_combine = false
```

The workflow downloads source data, runs rolling backtests, runs position-specific backtests, applies validation gates, and uploads a report artifact named `apex-backtest-reports`.

## Add production / consensus data

Create templates:

```bash
python src/build_features.py --write-templates
```

Fill any optional files under:

```text
data/production/
data/consensus/
```

Then build the enriched feature table:

```bash
python src/build_features.py
```

This writes:

```text
data/model_features.csv
reports/feature_coverage.json
```

## Retrain current production board

```bash
pip install -r requirements.txt
python src/download_source_data.py
python src/improve.py
```

This writes:

```text
data/apex_board.csv
models/apex_resid_*.txt
models/apex_baseline_and_transforms.pkl
reports/holdout_2012_2014_metrics.json
reports/holdout_2012_2014_scored.csv
```

## Rolling validation

```bash
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

This writes:

```text
reports/rolling_backtest_summary.csv
reports/rolling_backtest_by_position.csv
reports/rolling_backtest_report.json
```

The key number to watch is:

```text
delta_plus_vs_pick_spearman_drafted
```

Positive means APEX+ ranked drafted players better than the pick-only market baseline for that test year.

## Accuracy experiments

Feature-set experiment:

```bash
python src/experiment_feature_sets.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

Position-specific residual experiment:

```bash
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

Pre-draft forecasting experiment:

```bash
python src/predraft_backtest.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

Promotion gates:

```bash
python src/validation_gates.py reports/position_model_backtest_summary.csv
python src/validation_gates.py reports/feature_set_experiment_summary.csv --candidate-col feature_set
```

## Promotion rule

Promote a candidate only if it improves:

- average lift
- median lift
- window win rate
- worst-window behavior
- practical draft metrics such as precision@32 / precision@64

A single higher headline Spearman is not enough.

## Roadmap

Highest-impact next data additions:

1. consensus-board / expected-pick history for true pre-draft forecasting
2. QB pressure-to-sack and age-adjusted efficiency features
3. WR YPRR / target share / breakout age
4. EDGE pressure rate and pass-rush win rate
5. OL pressure allowed and snap data
6. DB coverage and missed-tackle metrics
