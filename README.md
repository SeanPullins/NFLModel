# APEX Draft Model

Two-stage residual NFL draft model + interactive scouting dashboard covering historical classes and forward-looking prospect watchlists.

The live v1.2.1 board reports **APEX+ Spearman ρ 0.697** on the 2012-14 official holdout versus **0.615** for the pick-only market baseline. Across the four published rolling validation windows from 2008-2020, APEX+ improves over the market in **4 of 4 windows**, with **average lift +0.083** and **median lift +0.080**.

## What changed in v1.2+

- **Fold-safe college encoding** — college strength is fit on the training fold only, then mapped to validation/test/prospect rows.
- **Fold-safe athletic normalization** — position z-scores are learned from the training fold only instead of the full dataframe.
- **Repo-relative paths** — scripts now read from `data/` or `APEX_DATA_DIR` instead of hardcoded scratch paths.
- **Rolling backtests** — `src/backtest.py` now reports raw APEX and APEX+ versus pick-only/market baselines.
- **APEX+ validation artifact** — committed CSV/JSON summaries show average and median lift across rolling windows.
- **Public validation page** — `docs/validation.html` shows the rolling-window scoreboard and links to downloadable CSV/JSON.
- **Model card** — assumptions, target definition, leakage controls, and known limitations are documented in `docs/MODEL_CARD.md`.

## Live dashboard

Deploy free on GitHub Pages:

```bash
git init && git add . && git commit -m "APEX v1.2.1"
gh repo create apex-draft-model --public --source=. --push
```

Then: repo **Settings → Pages → Source: main / `/docs`**.

The dashboard reads `data/apex_board.csv`, builds a searchable board, and displays pick-vs-outcome, surplus, and player-level scoring views.

Useful public pages:

- `docs/index.html` — board
- `docs/model.html` — model explanation
- `docs/validation.html` — rolling validation summary
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
  pipeline.py     shared loading, feature, baseline, residual, and metric utilities
  improve.py      v1.2 train + original 2012-14 holdout evaluation
  backtest.py     rolling out-of-time validation, including APEX+
  build_site.py   static dashboard builder
  template.html   dashboard template

data/
  apex_board.csv  generated board used by dashboard
  SOURCES.md      raw-data source notes

docs/
  index.html                     GitHub Pages dashboard
  model.html                     model explanation
  validation.html                rolling validation page
  rolling_backtest_summary.csv   public validation CSV
  rolling_backtest_report.json   public validation JSON
  MODEL_CARD.md                  model assumptions, limits, validation protocol
  VALIDATION.md                  validation guidance and interpretation

models/
  generated LightGBM/isotonic artifacts after running src/improve.py

reports/
  generated holdout and rolling backtest outputs
```

## Raw data setup

The training scripts expect these raw files:

```text
draft_data.csv
combine_data_pfr_with_stats.csv
```

Put them in `data/`, or point to another folder:

```bash
export APEX_DATA_DIR=/path/to/raw-data
```

Source notes are in `data/SOURCES.md`.

## Retrain

```bash
pip install -r requirements.txt
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
python src/backtest.py --first-test-year 2011 --last-test-year 2016 --apex-plus-factor 3.5
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

## Roadmap

Highest-impact next upgrades:

1. Add college production features by position.
2. Split pre-draft and post-draft models.
3. Add consensus-board / mock-draft expected-pick data for pre-draft forecasting.
4. Build QB-specific, WR-specific, OL-specific, EDGE-specific, and DB-specific submodels.
5. Add uncertainty bands and tier labels instead of only raw decimals.
6. Update the future board with current pick data and keep source generation reproducible.
