# QB-only scoring system: no-noise prospect model

Purpose: build a QB-specific prospect score from measurable college performance, advanced metrics, experience, and physical thresholds only. It intentionally excludes media buzz, mock-draft rank, comps, interviews, and consensus-board noise.

## Agent team

| Agent | Assignment | Allowed inputs | Output |
|---|---|---|---|
| QB Data Integrity Agent | Confirm every field is pre-draft and measurable. | PFF/CFBD/QBR/combine/draft-age fields | coverage report |
| Efficiency Context Agent | Measure play value, not raw volume. | EPA/play, QBR, PFF pass grade, PPA | efficiency score |
| Accuracy Agent | Measure ball placement/expected accuracy. | CPOE, adjusted completion %, charted accuracy | accuracy score |
| Pressure & Sack Agent | Separate sack creation/avoidance from OL noise as much as possible. | pressure-to-sack, sack rate, sack EPA, time to throw | pressure score |
| Decision Risk Agent | Penalize avoidable turnover profile. | turnover-worthy play rate, INT rate | risk score |
| Creation Agent | Reward high-end NFL throws and chain-moving. | big-time throw rate, first-down rate, positive EPA %, deep efficiency | creation score |
| Experience Curve Agent | Protect against tiny samples and old-prospect inflation. | starts, career plays/dropbacks, seasons, age | sample/age score |
| Mobility Agent | Add rushing/scramble value without letting it replace passing. | rush EPA, rush EPA/game, scramble rate | mobility score |
| Measurables Agent | Apply small threshold gates, not heavy bonuses. | height, weight, hand size | measurable score |
| Validation Agent | Promote nothing until rolling QB-only validation passes. | historical QB outcomes | promotion report |

## Score contract

The first score is **QB Trait Score**, a 0-100 no-noise trait model. It is not a final draft grade and not a depth-chart projection.

It answers:

> Did the prospect show the measurable college traits that historically belong in a serious NFL QB evaluation?

It does **not** answer:

- Will the NFL team develop him?
- Is he loved by coaches?
- Is he rising in mocks?
- Does a public analyst like him?
- Is the landing spot ideal?

## Component weights

| Component | Weight | Why it matters |
|---|---:|---|
| Efficiency/context | 22 | QB must create value per play, not just totals. QBR/EPA-style metrics are preferred because they can incorporate sacks, rushes, difficulty, opponent/context, and garbage-time adjustments when available. |
| Accuracy | 16 | Ball placement travels better than raw completion percentage when adjusted for throw difficulty and drops. |
| Pressure/sack avoidance | 16 | Sacks and pressure-to-sack behavior are QB-influenced enough to be a major risk area. |
| Decision risk | 14 | Turnover-worthy plays and INT rate identify reckless or late decision profiles. |
| Creation/explosiveness | 12 | NFL starters need high-end throws, first-down creation, and positive-play production. |
| Experience/sample | 8 | Tiny samples are dangerous; age and starts need to be treated as context. |
| Mobility | 7 | Rushing/scramble value raises floor and creates answers when structure breaks, but cannot replace passing. |
| Measurable thresholds | 5 | Size/hand/frame concerns matter, but should be small gates rather than the model's core. |

## Metric families

### Efficiency/context

Primary fields:

- `qb_final_qbr`
- `qb_best_qbr`
- `qb_epa_per_play`
- `qb_final_epa_per_play`
- `qb_pff_pass_grade`
- `qb_pff_pass_grade_final`
- future: CFBD player PPA, opponent-adjusted PPA, garbage-time-filtered PPA

### Accuracy

Primary fields:

- `qb_cpoe`
- `qb_adj_completion_pct`
- `qb_accuracy_percent`
- `qb_final_accuracy_percent`

Avoid using raw completion percentage as a major driver unless no adjusted accuracy field exists.

### Pressure/sack avoidance

Primary fields:

- `qb_pressure_to_sack_rate`
- `qb_sack_rate`
- `qb_sack_epa_per_play`
- `qb_avg_time_to_throw` as a light context field

Rules:

- High pressure-to-sack is a risk flag.
- Low sack rate with bad efficiency is not enough.
- Time to throw is a weak feature unless paired with pressure/sack outcomes.

### Decision risk

Primary fields:

- `qb_turnover_worthy_play_rate`
- `qb_interception_rate`
- `qb_final_twp_rate`

Rules:

- Prefer TWP over INTs when available.
- INT rate is noisy and must not overrule stronger charted risk data.

### Creation/explosiveness

Primary fields:

- `qb_big_time_throw_rate`
- `qb_first_down_rate`
- `qb_positive_epa_percent`
- `qb_deep_ypa`

Rules:

- Big-time throws without risk control are volatile.
- Low creation plus high sack/pressure risk is a major red flag.

### Experience/sample

Primary fields:

- `qb_career_plays`
- `qb_starts`
- `qb_seasons`
- `age`

Risk flags:

- `very_small_sample`: fewer than 450 career QB action plays/dropbacks
- `small_sample`: fewer than 800 career plays/dropbacks
- `older_prospect`: draft age over 24

### Mobility

Primary fields:

- `qb_run_epa_per_play`
- `qb_rush_epa_per_game`
- `qb_scramble_rate`

Rule: mobility is additive. It cannot cover up a bad passing profile.

### Measurables

Primary fields:

- `height`
- `weight`
- `hand_size`

Rules:

- Treat as thresholds and flags.
- Do not make size a major positive driver.
- Apply small penalties for extreme outliers only.

## Score tiers

| QB Trait Score | Tier |
|---:|---|
| 85+ | blue_chip_trait_profile |
| 75-84 | starter_trait_profile |
| 65-74 | developmental_starter_traits |
| 55-64 | backup_or_system_profile |
| <55 | low_trait_score |

## Hard no-noise exclusions

The first QB score excludes:

- mock draft rank
- media consensus
- Twitter buzz
- comparison/player comp text
- unnamed scout quotes
- interview rumors
- team-needs speculation
- landing spot

A separate post-draft/draft-capital prior can be built later, but it must not contaminate the pure QB trait score.

## Validation plan

1. Run `src/qb_scoring_system.py` on historical QB production files.
2. Join to mature NFL outcomes.
3. Test against three outcome targets:
   - starting seasons
   - weighted AV / career value where available
   - bust/floor outcome
4. Compare to:
   - pick-only baseline
   - age + pick baseline
   - APEX generic QB output
5. Promote only if rolling out-of-time QB-only validation passes with acceptable worst-year drawdown.

## Command

```bash
python src/qb_scoring_system.py \
  --input data/production/qb_production.csv \
  --out reports/qb_trait_scores.csv \
  --report reports/qb_scoring_system_report.json
```

## Current warning

QB is the one position where a generic all-position score is most likely to lie. This score is a QB-only research product until it validates on historical QB outcomes.
