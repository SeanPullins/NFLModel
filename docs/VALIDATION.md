# Validation Guide

## Current honest status

The regenerated public-source run showed that the old **APEX+ 3.5×** headline should not be promoted yet. Raw APEX showed a small edge over draft slot on the regenerated 2011-2016 run, while 3.5× amplification hurt performance. The next validation run now fixes the hidden year cap and tests 2011-2021 with `--end-year 2021`.

Current claim:

> Raw APEX is the honest headline model until an APEX+ residual factor passes promotion gates.

Avoid claiming:

> APEX+ 3.5× is proven to beat the market.

## What changed

The old scripts silently loaded only through 2016 because `load_dataset()` defaulted to `end_year=2016`. The backtest scripts now expose `--end-year`, and the GitHub Action now runs:

```bash
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --apex-plus-factor 3.5
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --apex-plus-factor 3.5
```

## Main outputs

```text
reports/rolling_backtest_summary.csv
reports/rolling_backtest_by_position.csv
reports/rolling_backtest_report.json
reports/apex_factor_sweep_summary.csv
reports/apex_factor_sweep_by_year.csv
reports/apex_factor_sweep_report.json
reports/position_model_backtest_summary.csv
reports/position_model_backtest_report.json
reports/rolling_validation_gates.json
reports/position_model_validation_gates.json
```

## Main metrics

For raw APEX:

```text
delta_raw_vs_pick_spearman_drafted
```

For an APEX+ factor:

```text
delta_plus_vs_pick_spearman_drafted
```

Interpretation:

| Value | Meaning |
|---:|---|
| Positive | Model ranked drafted players better than draft slot alone |
| Near zero | Model roughly matched the market |
| Negative | Model underperformed draft slot |

## APEX+ factor sweep

APEX+ uses:

```text
APEX+ = market + factor × (raw_apex - market)
```

Factor meanings:

| Factor | Meaning |
|---:|---|
| 0.0 | market-only baseline |
| 1.0 | raw APEX |
| >1.0 | residual amplification |

A factor is promoted only if it passes validation gates. If no factor passes, the public headline remains raw APEX.

## Promotion gates

A model should not be promoted unless it improves:

1. positive average lift over pick-only
2. positive median lift over pick-only
3. adequate window win rate
4. acceptable worst-window loss
5. practical draft metrics such as precision@32 or precision@64 when available

The default gate file is:

```text
src/validation_gates.py
```

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

## Recommended wording

Use:

> Raw APEX is being validated as a residual-over-market draft model. APEX+ amplification is experimental and must pass rolling-window promotion gates before it becomes the headline model.

Do not use:

> APEX+ 3.5× is the proven best model.
