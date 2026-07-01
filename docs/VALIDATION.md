# Validation Guide

## Why validation changed

The original APEX result used a useful but narrow test:

- train mostly on older classes
- evaluate 2012-14
- compare APEX against pick-only

That was a good prototype test, but one holdout window is not enough to know whether the model has a durable edge. v1.2 adds rolling out-of-time validation.

## Commands

Run the original holdout-style evaluation:

```bash
python src/improve.py
```

Run rolling validation:

```bash
python src/backtest.py --first-test-year 2011 --last-test-year 2016
```

Use newer raw data:

```bash
APEX_DATA_DIR=/path/to/raw python src/backtest.py --first-test-year 2011 --last-test-year 2021
```

## Main outputs

```text
reports/holdout_2012_2014_metrics.json
reports/holdout_2012_2014_scored.csv
reports/rolling_backtest_summary.csv
reports/rolling_backtest_by_position.csv
reports/rolling_backtest_report.json
```

## Main metric

The most important column is:

```text
delta_apex_vs_pick_spearman_drafted
```

Interpretation:

| Value | Meaning |
|---:|---|
| Positive | APEX ranked drafted players better than draft slot alone |
| Near zero | APEX roughly matched the market |
| Negative | APEX underperformed draft slot |

## What would count as a stronger model

APEX becomes meaningfully more convincing if rolling validation shows:

1. positive average lift over pick-only
2. positive median lift over pick-only
3. position-level lift that repeats across years
4. better precision@32 or precision@64
5. better bust avoidance among highly drafted players
6. confidence intervals that are not centered near zero

## How to avoid false confidence

Do not rely only on the dashboard score.

For any player, check:

- position
- pick or expected pick
- model surplus
- whether that position has shown historical lift
- whether the score is driven by athleticism only
- whether the model has production data for that player
- whether the player belongs to a high-volatility position like QB

## Current recommended claim

Until rolling validation is rerun and reviewed, the safest claim is:

> APEX is a residual draft model with an original 2012-14 holdout lift over pick-only. v1.2 improves validation hygiene and adds rolling backtests to test whether that edge is durable.

Avoid claiming:

> APEX consistently beats NFL teams.

That may become true in narrow spots, but it needs stronger evidence.
