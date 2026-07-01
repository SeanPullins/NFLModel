# APEX Feature Upgrade Guide

This guide explains how to add the next accuracy layer without contaminating validation.

## Goal

APEX currently gets most of its signal from draft capital, athletic/profile data, age, and college context. The next real accuracy gain should come from:

1. position-specific college production
2. pre-draft consensus/expected-pick data
3. position-specific residual models
4. promotion gates that prevent overfit upgrades

## File layout

Optional input files live here:

```text
data/production/
  qb_production.csv
  wr_production.csv
  rb_production.csv
  te_production.csv
  ol_production.csv
  edge_production.csv
  dt_production.csv
  lb_production.csv
  db_production.csv

data/consensus/
  consensus_board.csv
```

Create empty templates:

```bash
python src/build_features.py --write-templates
```

Then fill the CSVs and rebuild:

```bash
python src/build_features.py
```

That writes:

```text
data/model_features.csv
reports/feature_coverage.json
```

## Required keys

Every optional CSV needs:

```text
Year,Player
```

Recommended columns:

```text
Pos,College
```

The merge is fuzzy-light, not fuzzy-heavy: player names are normalized by removing punctuation/suffixes and merging on `normalized_name + Year`.

## Production feature examples

### QB

```text
qb_starts
qb_age_adj_epa_per_play
qb_cpoe
qb_adj_completion_pct
qb_pressure_to_sack_rate
qb_sack_rate
qb_big_time_throw_rate
qb_turnover_worthy_play_rate
qb_rush_epa_per_game
```

### WR

```text
wr_yards_per_route_run
wr_target_share
wr_dominator
wr_breakout_age
wr_explosive_reception_rate
wr_contested_catch_rate
wr_slot_rate
```

### OL / defense

```text
ol_pressure_rate_allowed
ol_blown_block_rate
edge_pressure_rate
dt_run_stop_rate
lb_missed_tackle_rate
db_yards_per_coverage_snap
```

## Consensus feature examples

```text
consensus_rank
expected_pick
mock_avg_pick
mock_pick_std
n_mocks
n_big_boards
nfl_com_grade
recruiting_stars
recruiting_rank
combine_invite
senior_bowl
shrine_bowl
```

## Validation workflow

### 1. Rebuild enriched table

```bash
python src/build_features.py
```

### 2. Test the current post-draft model

```bash
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

### 3. Test feature-set changes

```bash
python src/experiment_feature_sets.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

### 4. Test position-specific residual models

```bash
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

### 5. Test true pre-draft forecasting

```bash
python src/predraft_backtest.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

### 6. Apply promotion gates

```bash
python src/validation_gates.py reports/position_model_backtest_summary.csv
python src/validation_gates.py reports/feature_set_experiment_summary.csv --candidate-col feature_set
```

## Promotion rule

Do not promote a change unless it improves:

- average lift
- median lift
- win rate
- worst-window behavior
- at least one practical metric such as precision@32 or precision@64

A higher headline Spearman in one window is not enough.

## Best next data sources to add

Highest priority:

1. consensus expected pick / big board history
2. QB production and pressure/sack traits
3. WR YPRR / target share / breakout age
4. EDGE pressure rate and pass-rush win rate
5. OL pressure allowed and snap data

Best first target: consensus expected pick, because it unlocks a true pre-draft model.
