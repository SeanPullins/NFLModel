# APEX Draft Model Card

## Purpose

APEX is a draft-market residual model. It is designed to answer:

> Which prospects look better or worse than the historical value implied by their draft slot?

It is not designed to produce a precise estimate of a player's final Career AV, and it should not be used as a standalone scouting grade.

## Target

The model predicts **within-class Career AV percentile**.

Why this target exists:

- Raw Career AV varies by draft class.
- Percentile ranking makes each class comparable.
- It reduces, but does not eliminate, career-length and era effects.

Known weakness:

- Career AV is an imperfect player-quality measure.
- It favors durable players and positions that accumulate AV more easily.
- Recent classes can be heavily censored and should not be used as mature outcome tests too early.

## Features

Current residual features:

- position-normalized athletic testing
- speed score
- explosion
- agility
- height
- weight
- BMI
- bench
- age, when present
- shrunken historical college encoding

Market features:

- actual draft pick for post-draft scoring
- isotonic pick-to-outcome expectation
- optional per-position pick curve blend

## Validation protocol

The upgraded v1.2 code uses out-of-time validation:

1. Fit transforms and models on past classes only.
2. Tune residual shrinkage on validation classes.
3. Refit on pre-test classes.
4. Score the unseen test class or test window.
5. Compare APEX against:
   - pick-only baseline
   - position-blended pick baseline

Primary metric:

- Spearman correlation on drafted players

Secondary metrics:

- hit AUC
- precision@32
- precision@64
- NDCG@32
- NDCG@64
- position-level Spearman

## Leakage controls added in v1.2

### College encoding

College strength is now fit on the training fold only and mapped onto validation/test rows. This prevents the model from seeing future school outcomes while evaluating past forecasts.

### Athletic z-scores

Position-normalized athletic z-scores are now fit on the training fold only. Validation/test combine distributions no longer influence their own normalization.

### File paths

Training no longer depends on `/home/claude/...` scratch paths. Raw files can live in `data/` or a folder supplied by `APEX_DATA_DIR`.

## Interpretation

A positive residual does **not** mean a player is guaranteed to be good.

It means:

> Given the available market and profile information, this player looks better than the average historical player with a similar pick/position profile.

The most useful model outputs are likely:

- big disagreements with the market
- position-level edges
- late-round surplus flags
- high-pick bust warnings
- groups of comparable historical profiles

## Known limitations

1. **Production data does not clear the bar.**  
   NCAA box-score production features (`profile_plus_production` and related ablations) were tested against the profile-only default across the same rolling 2011-2021 window and did not beat it on average lift, median lift, win rate, and worst-window behavior. They remain available for experiments but are not part of the public board.

2. **Real pre-draft consensus data has now been tested, and it does not beat the market either.**  
   `src/download_consensus_data.py` pulls ESPN's actual pre-draft overall rank, position rank, and grade (from `JackLich10/nfl-draft-data`, ~96% match coverage on 2011-2021 drafted players) into `data/consensus/consensus_board.csv`, and `src/predraft_backtest.py` uses it to run a genuine pre-draft-only forecast (no actual draft slot as input). Result: ESPN's pre-draft rank alone is a **weaker** predictor of career outcome than the real draft market in every one of the 6 fully-covered rolling test years (2011-2016), by a wide margin in some years (e.g. 2016: 0.446 vs. 0.581 Spearman). Adding ESPN grade/position-rank as extra features on top of the existing post-draft profile model does not help either (mean delta ~0, 50% win rate over 2011-2016). Read literally: a single outlet's public pre-draft big board carries less information than the aggregate of 32 teams' actual draft decisions, and once real draft slot is known, ESPN's grade adds nothing on top of it. A true pre-draft model that beats the post-draft model would need non-public inputs (team medical grades, private workouts, character/makeup interviews) that are out of scope for this project's public data sources.

3. **QB needs a separate model.**  
   Generic combine/profile features are not enough for quarterback forecasting.

4. **Career AV is noisy.**  
   It is a useful public target, but it is not a complete measure of player quality.

5. **Small headline lifts can be fragile, and the per-position shrinkage tuner can overfit thin validation slices.**  
   Rolling-window testing found the residual shrinkage tuner (which picks a 0-1 weight per position from just a 2-year validation slice) would occasionally pick weights near 1.0 off a handful of players, then blow up out-of-sample - most visibly DB in the 2011 test year (shrink=0.9, then -0.08 Spearman versus pick-only). Capping the search at 0.4 (`src/pipeline.py::tune_position_shrinkage`) raised mean lift over pick-only from +0.012 to +0.015 Spearman, raised win rate from 73% to 82%, and tightened the worst rolling window from -0.033 to -0.018 - which is also what lets the model clear its own promotion gate (`src/validation_gates.py`) for the first time. APEX should still be judged over rolling backtests, not one holdout window.

6. **Future watchlist rows need a reproducible generator.**  
   The dashboard can contain future prospects, but source-of-truth generation should be scripted and versioned.

## Recommended next model version

Tested and not promoted (see Known limitations above): consensus big board / expected draft slot, college production metrics. Both are available for ablations via `--feature-set` and `src/predraft_backtest.py`, but neither beat the profile-only default.

APEX v2 should instead prioritize:

- a dedicated QB submodel
- position-specific submodels validated on worst-window behavior, not just mean lift (the existing `position_profile_only` challenger has this exact problem)
- uncertainty intervals
- tier labels
- draft-year freshness checks
- mature-outcome-only validation gates
- non-public inputs if ever available (team medical grades, private workout data) - the public data sources tested so far (combine profile, NCAA production, ESPN pre-draft consensus) are close to exhausted
