# APEX Draft Model

Two-stage residual NFL draft model + interactive scouting dashboard covering historical classes and forward-looking prospect watchlists.

The original v1.1 holdout reported **Spearman ρ 0.630** on 2012-14 drafted players versus **0.619** for a pick-only market baseline. v1.2 keeps the same residual-over-market idea, but upgrades the training code so validation is cleaner and repeatable.

## What changed in v1.2

- **Fold-safe college encoding** — college strength is fit on the training fold only, then mapped onto validation/test/prospect rows.
- **Fold-safe athletic normalization** — position z-scores are learned from the training fold only instead of the full dataframe.
- **Repo-relative paths** — scripts now read from `data/` or `APEX_DATA_DIR` instead of hardcoded scratch paths.
- **Rolling backtests** — added `src/backtest.py` to test year-by-year instead of relying on one 2012-14 window.
- **Validation artifacts** — holdout and rolling validation outputs write to `reports/`.
- **Model card** — assumptions, target definition, leakage controls, and known limitations are documented in `docs/MODEL_CARD.md`.

## Live dashboard

Deploy free on GitHub Pages:

```bash
git init && git add . && git commit -m "APEX v1.2"
gh repo create apex-draft-model --public --source=. --push
```

Then: repo **Settings → Pages → Source: main / `/docs`**.

The dashboard reads `data/apex_board.csv`, builds a searchable board, and displays pick-vs-outcome, surplus, and player-level scoring views.

## Architecture

1. **Market baseline** — isotonic regression from pick → outcome, with optional per-position blending.
2. **Athletic residual** — 5-seed bagged LightGBM on position-normalized combine/profile features, age, and shrunken college encoding.
3. **Per-position shrinkage** — residual weight tuned by position on an earlier validation fold, then applied to the out-of-time test fold.

**Target:** within-class Career AV percentile. This is a ranking target, not a calibrated projection of exact career value.

**Important interpretation:** APEX should be judged by whether it improves the rank ordering of prospects over the draft market. A small Spearman lift can be useful, but it is not proof that the model consistently beats NFL teams.

## Repo layout

```text
src/
  pipeline.py     shared loading, feature, baseline, residual, and metric utilities
  improve.py      v1.2 train + original 2012-14 holdout evaluation
  backtest.py     rolling out-of-time validation
  build_site.py   static dashboard builder
  template.html   dashboard template

data/
  apex_board.csv  generated board used by dashboard
  SOURCES.md      raw-data source notes

docs/
  index.html      GitHub Pages dashboard
  MODEL_CARD.md   model assumptions, limits, validation protocol
  VALIDATION.md   validation guidance and interpretation

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
python src/backtest.py --first-test-year 2011 --last-test-year 2016
```

This writes:

```text
reports/rolling_backtest_summary.csv
reports/rolling_backtest_by_position.csv
reports/rolling_backtest_report.json
```

The key number to watch is:

```text
delta_apex_vs_pick_spearman_drafted
```

Positive means APEX ranked drafted players better than the pick-only market baseline for that test year.

## Roadmap

Highest-impact next upgrades:

1. Add college production features by position.
2. Split pre-draft and post-draft models.
3. Add consensus-board / mock-draft expected-pick data for pre-draft forecasting.
4. Build QB-specific, WR-specific, OL-specific, EDGE-specific, and DB-specific submodels.
5. Add uncertainty bands and tier labels instead of only raw decimals.
6. Update the future board with current pick data and keep source generation reproducible.
