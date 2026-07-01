# Validation Guide

## Current honest status

The public/default model is now **profile-only raw APEX**:

```text
draft market baseline + combine/profile features + age + college encoding
```

The old **APEX+ 3.5x** headline is not promoted. NCAA production features and real ESPN pre-draft consensus data (`data/consensus/consensus_board.csv`, via `src/download_consensus_data.py`) are also not promoted into the public board, because rolling-window testing did not show a strong enough improvement over profile-only models - see `docs/MODEL_CARD.md` for the actual numbers.

**Real numbers, regenerated from a live rolling 2011-2021 backtest** (`python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile`), not copied from an old holdout page:

| Metric | Value |
|---|---:|
| Mean lift vs. pick-only (Spearman, drafted players) | +0.0146 |
| Median lift | +0.0181 |
| Win rate (11 rolling test years) | 82% (9/11) |
| Worst rolling window | -0.0177 (2011) |
| 95% bootstrap CI on mean lift | [+0.0046, +0.0235] |
| `src/validation_gates.py` result | **pass** |

This reflects a fix landed in this pass: the per-position residual shrinkage tuner (`tune_position_shrinkage`) used to search up to a 1.0 weight off a thin 2-year validation slice, which let it occasionally overfit (DB picked 0.9 in the 2011 fold, then lost badly out-of-sample). Capping the search at 0.4 raised the mean lift from +0.012 to +0.015, tightened the worst window from -0.033 to -0.018, and is what lets the model pass its own gate for the first time - previously, running the actual gate script against a real (not copied) backtest showed the public default **failing** its own bar.

Current claim:

> APEX finds a small, repeatable edge over draft slot (mean +0.015 Spearman lift, 82% rolling-window win rate, positive 95% CI). The public board uses profile-only raw APEX with capped position shrinkage. APEX+ amplification, NCAA production, and real ESPN pre-draft consensus data all remain tested-but-not-promoted.

Avoid claiming:

> APEX+ 3.5x is proven to beat the market.

Avoid claiming:

> NCAA box-score production improves the headline model.

Avoid claiming:

> Consensus/expected-pick data would fix the "no real pre-draft model" gap - it was tested with real ESPN data and it doesn't.

## Model roles

| Component | Status | Use |
|---|---|---|
| Global profile-only raw APEX, capped position shrinkage | Public/default board | Main score, passes `validation_gates.py` |
| Position profile-only raw APEX | Top challenger | Track in reports |
| NCAA production features | Tested, not promoted | Ablation only |
| Real ESPN pre-draft consensus rank/grade | Tested, not promoted | `predraft_backtest.py` only |
| APEX+ residual amplification | Tested, not promoted | Factor sweep only |

## Current workflow commands

The workflow builds a broad board through 2026 but validates mature outcome windows through 2021:

```bash
python src/download_source_data.py
python src/improve.py --feature-set profile --end-year 2026
python src/build_site.py
python src/backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile --apex-plus-factor 3.5
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year 2021 --end-year 2021 --feature-set profile --apex-plus-factor 3.5
python src/ablation_backtest.py --first-test-year 2011 --last-test-year 2021 --end-year 2021
python src/validation_gates.py reports/rolling_backtest_summary.csv --delta-col delta_raw_vs_pick_spearman_drafted --out reports/rolling_validation_gates.json
python src/validation_gates.py reports/position_model_backtest_summary.csv --delta-col delta_raw_vs_pick_spearman_drafted --out reports/position_model_validation_gates.json
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
reports/feature_ablation_summary.csv
reports/feature_ablation_by_year.csv
reports/feature_ablation_report.json
reports/rolling_validation_gates.json
reports/position_model_validation_gates.json
data/apex_board.csv
index.html
docs/index.html
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

For feature ablations:

```text
delta_candidate_vs_pick_spearman_drafted
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
APEX+ = market + factor x (raw_apex - market)
```

Factor meanings:

| Factor | Meaning |
|---:|---|
| 0.0 | market-only baseline |
| 1.0 | raw APEX |
| >1.0 | residual amplification |

A factor is promoted only if it passes validation gates and improves on raw APEX. If no factor passes, the public headline remains raw APEX.

## Feature ablation rule

A production feature family should not be promoted unless it beats profile-only on:

1. average lift
2. median lift
3. win rate
4. worst-window behavior
5. practical metrics such as precision@32 or precision@64 when available

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
- whether the score is driven by athleticism/profile data only
- whether the player belongs to a high-volatility position like QB

## Recommended wording

Use:

> Raw profile-only APEX is the public residual-over-market draft model. APEX+ amplification and NCAA production are experimental and must pass rolling-window promotion gates before becoming headline inputs.

Do not use:

> APEX+ 3.5x is the proven best model.
