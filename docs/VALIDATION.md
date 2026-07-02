# Validation Guide

## Current honest status

The public/default model is now **profile-only raw APEX**:

```text
draft market baseline + combine/profile features + age + college encoding
```

The old **APEX+ 3.5x** headline is not promoted. NCAA production features are also not promoted into the public board because ablation testing did not show a strong enough improvement over profile-only models.

Current claim:

> APEX finds a small, repeatable edge over draft slot. The public board uses profile-only raw APEX. APEX+ amplification and NCAA production remain experimental.

Avoid claiming:

> APEX+ 3.5x is proven to beat the market.

Avoid claiming:

> NCAA box-score production improves the headline model.

## Model roles

| Component | Status | Use |
|---|---|---|
| Global profile-only raw APEX | Public/default board | Main score |
| Position profile-only raw APEX | Top challenger | Track in reports |
| NCAA production features | Experimental | Ablation only |
| APEX+ residual amplification | Experimental | Factor sweep only |
| Pre-draft APEX (ESPN consensus baseline) | Measured, no edge | `predraft_backtest.py` report only |
| Consensus board-vs-pick features | Experimental | `profile_plus_consensus` feature set |

## Pre-draft validation result (2011-2021)

With ESPN pre-draft boards (2004-2021) as the market proxy:

| Forecaster | Mean Spearman (drafted) |
|---|---:|
| Actual draft slot | ~0.59 |
| ESPN consensus rank | ~0.52 |
| Pre-draft APEX raw | ~0.52 (delta vs consensus: mean -0.004, median +0.002, win rate 6/11) |
| Pre-draft APEX+ 3.5x | loses all 11 years — never use amplification pre-draft |

Honest claim: the model matches public consensus pre-draft but has not beaten
it, and nobody public beats the actual draft order on average.

## Consensus board-vs-pick experiment (post-draft, 2011-2021)

`profile_plus_consensus` (profile + consensus rank + ESPN grade +
board-vs-pick disagreement):

| Metric | profile (default) | profile_plus_consensus |
|---|---:|---:|
| Mean lift vs pick | +0.0117 | +0.0076 |
| Median lift | +0.0138 | +0.0160 |
| Win rate | 8/11 | 8/11 |
| Worst year | -0.0328 | -0.0583 |

Verdict: better median, worse mean and worst-window. Fails promotion rules;
stays experimental.

## College QBR experiment (position-family challenger, 2011-2021)

`src/build_qb_production.py` adds ESPN college Total QBR features (2004+) to
the QB family inside the position-family challenger model:

| Metric | position profile-only | + QB college QBR |
|---|---:|---:|
| Mean lift vs pick | +0.0131 | +0.0139 |
| Median lift | +0.0171 | +0.0171 |
| Win rate | 9/11 | 10/11 |
| Worst year | -0.0515 | -0.0515 |

QBR data only reaches 25% training coverage from the 2016 test year onward;
every year it is active improves or ties. Kept in the challenger configuration.
The challenger still fails the strict worst-window gate (2011), so the global
profile-only model remains the public default.

## Bust-avoidance stress test (2011-2021)

Bust = drafted-peer career percentile < 0.45 among top-100 picks (base rate
~20%). Rolling out-of-time backtest:

| Metric | Slot-implied odds | Trait model |
|---|---:|---:|
| Mean AUC | 0.702 | 0.713 |
| Top-10 flag precision | 34% | 37% |
| AUC win years | - | 6/11 |

Strongest red-flag traits on all mature classes (bust rate, worst vs best
quintile): consensus reach 33%/4%, ESPN grade 33%/4%, age 23+ 26%/13%,
weak college pedigree 27%/16%, poor agility 23%/13%. See
`reports/bust_trait_table.csv`.

## PFF QB features: first measurement (2026-07, partial seasons)

With PFF passing-page exports for 11 of 12 seasons (2014-2022, 2024, 2025;
only 2023 missing) mapped into QB features and the recent-window coverage
rule active:

| Challenger config | Mean lift | Wins | Worst |
|---|---:|---:|---:|
| position profile-only | +0.0131 | 9/11 | -0.0515 |
| + ESPN college QBR | **+0.0139** | **10/11** | -0.0515 |
| + QBR + PFF (11 seasons) | +0.0132 | 9/11 | -0.0515 |

Identical verdict at 6 and 11 seasons, and the reason is structural: mature
test years end at 2021, so the backtest can only ever see PFF for the
2015-2020 QB classes (~13 drafted QBs per year). In that thin window PFF
duplicates the QBR signal and adds noise (2020 flips negative, 2021 -0.003).
Most of PFF's value sits on the 2022-2026 classes, whose outcomes are still
censored - it cannot be validated yet, only shipped as clearly-labeled
projection input. Decision: promoted challenger stays QBR-only; PFF QB
features remain in the data layer and can inform experimental projections
for new classes. Re-measure when the 2022 class matures (~2027).

## Season-trajectory features: investigated, not implemented (2026-07)

Public season-by-season college stats (JackLich10 `college_statistics.csv` and
`sportsdataverse/cfbfastR-data`, both 2014+) cover draft classes 2019-2021 at
~86% but classes before 2018 at ~0%. That leaves no honest out-of-time test:
covered classes lack covered training history and have censored outcomes.
Decision recorded in `reports/trajectory_feasibility.json`; revisit when the
2019-2021 classes mature (~2027).

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
