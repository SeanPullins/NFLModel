# Validation Guide

## Why validation changed

The original APEX result used a useful but narrow test:

- train mostly on older classes
- evaluate 2012-14
- compare APEX against pick-only

That was a good prototype test, but one holdout window is not enough to know whether the model has a durable edge. v1.2 added rolling out-of-time validation; v1.2.1 adds APEX+ to that validation output and publishes a downloadable summary.

## Commands

Run the original holdout-style evaluation:

```bash
python src/improve.py
```

Run rolling validation with the published APEX+ residual factor:

```bash
python src/backtest.py --first-test-year 2011 --last-test-year 2016 --apex-plus-factor 3.5
```

Use newer raw data:

```bash
APEX_DATA_DIR=/path/to/raw python src/backtest.py --first-test-year 2011 --last-test-year 2021 --apex-plus-factor 3.5
```

## Main outputs

```text
reports/holdout_2012_2014_metrics.json
reports/holdout_2012_2014_scored.csv
reports/rolling_backtest_summary.csv
reports/rolling_backtest_by_position.csv
reports/rolling_backtest_report.json
```

Public GitHub Pages validation artifacts:

```text
docs/validation.html
docs/rolling_backtest_summary.csv
docs/rolling_backtest_report.json
```

## Main metric

The most important generated column is:

```text
delta_plus_vs_pick_spearman_drafted
```

Interpretation:

| Value | Meaning |
|---:|---|
| Positive | APEX+ ranked drafted players better than draft slot alone |
| Near zero | APEX+ roughly matched the market |
| Negative | APEX+ underperformed draft slot |

## Published rolling-window summary

| Validation window | Market | Raw APEX | APEX+ | APEX+ lift |
|---|---:|---:|---:|---:|
| 2008-2011 | 0.585 | 0.625 | 0.689 | +0.104 |
| 2012-2014 official holdout | 0.615 | 0.658 | 0.697 | +0.082 |
| 2015-2017 | 0.596 | 0.637 | 0.674 | +0.078 |
| 2018-2020 | 0.601 | 0.641 | 0.670 | +0.069 |

Across those four published windows, APEX+ has:

- average lift over market: **+0.083 Spearman**
- median lift over market: **+0.080 Spearman**
- window win rate: **4 / 4**

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

The safest claim is:

> APEX+ improves over the draft-market baseline across the four published rolling validation windows, with average lift +0.083 and median lift +0.080 Spearman. That is stronger than a single holdout claim, but the report should still be regenerated from raw data after any dataset or feature change.

Avoid claiming:

> APEX consistently beats NFL teams in all future drafts.

That may become true in narrow spots, but it still requires future locked-class evidence.
