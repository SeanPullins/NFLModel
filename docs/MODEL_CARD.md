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

1. **Production data is missing.**  
   The current model is too reliant on athletic/profile data. Position-specific production should be the next major feature family.

2. **Actual draft pick is unavailable pre-draft.**  
   A true pre-draft model needs consensus rank or expected draft position.

3. **QB needs a separate model.**  
   Generic combine/profile features are not enough for quarterback forecasting.

4. **Career AV is noisy.**  
   It is a useful public target, but it is not a complete measure of player quality.

5. **Small headline lifts can be fragile.**  
   APEX should be judged over rolling backtests, not one holdout window.

6. **Future watchlist rows need a reproducible generator.**  
   The dashboard can contain future prospects, but source-of-truth generation should be scripted and versioned.

## Recommended next model version

APEX v2 should add:

- consensus big board / expected draft slot
- college production metrics
- position-specific submodels
- uncertainty intervals
- tier labels
- draft-year freshness checks
- mature-outcome-only validation gates
