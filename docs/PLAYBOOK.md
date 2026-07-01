# APEX Annual Playbook — how to rerun this for a new draft year

This is the operational "how" behind the model: what data goes in, what each
stage does, how the validation decides what is trustworthy, and exactly what to
run each year.

## 1. The idea in one paragraph

NFL front offices collectively set a market price for every prospect: the draft
slot. That market is very good — pick number alone correlates ~0.58-0.61
(Spearman) with career outcome within a class. APEX does not try to replace the
market; it models the *residual*: given everything publicly measurable about a
prospect (athletic testing, size, age, school history, pre-draft consensus
rank), does this player look better or worse than the average historical player
picked at the same slot? The final score is `market expectation + shrunken
residual`. Every claimed edge must survive rolling out-of-time backtests where
the model only ever sees the past.

## 2. Data inputs (all public, all scripted)

| Input | Source | Script | Role |
|---|---|---|---|
| Draft results + Career AV + NCAA stats (1987-2024) | `phcs971/nfl-draft-dataset` | `src/download_source_data.py` | outcomes + production features |
| Combine / pro-day measurements (1987-2026) | `array-carpenter/nfl-draft-data` | `src/download_source_data.py` | measurement overlay |
| ESPN pre-draft board: overall rank, position rank, scouting grade (2004-2021) | `JackLich10/nfl-draft-data` | `src/build_consensus_board.py` | pre-draft market proxy |

Nothing from a player's NFL career is ever used as a model input — NFL stats
exist in the sources but are outcomes, not features.

## 3. Model architecture (post-draft board)

1. **Target** — within-class Career AV percentile (`y`). Ranking target, not a
   point estimate of career value.
2. **Market baseline** — isotonic regression from draft pick to `y`, fit on
   training years only, blended 50/50 with per-position isotonic curves where
   a position has 300+ training samples.
3. **Residual model** — 5-seed bagged LightGBM predicting `y - baseline` from
   position-normalized athletic z-scores, age, and a shrunken college encoding
   (the `profile` feature set). All normalizations and encodings are fit on
   the training fold only.
4. **Per-position shrinkage** — the residual weight (0 to 1) is tuned per
   position on a validation fold that precedes the test year, because the
   residual is more trustworthy for some positions (e.g. RB/WR) than others
   (e.g. QB).
5. **Score** — `baseline + shrink[pos] * residual`.

## 4. Pre-draft variant (no draft-night information)

Same residual machinery, but the market baseline is fit on **ESPN pre-draft
consensus rank** instead of the actual pick, and ESPN grade / position rank are
available as residual features. This is the honest "can we out-forecast the
front offices before the draft happens" test — the actual pick is used only to
filter to drafted players and to compare against.

## 5. Validation protocol (the part that keeps us honest)

For each test year Y in 2011-2021:

- fit transforms + models on years < Y-2, tune shrinkage on years Y-2..Y-1;
- refit on all years < Y, score year Y untouched;
- compare against the pick-only baseline on drafted players.

Primary metric: `delta_raw_vs_pick_spearman_drafted` (model Spearman minus
pick-only Spearman). Secondary: hit AUC, precision@32/64, NDCG@32/64.

**Promotion gates** (`src/validation_gates.py`): a candidate model replaces the
public default only if it improves average lift, median lift, window win rate,
and worst-window loss. A single good headline year is never enough. Career AV
for classes after ~2021 is still censored — do not extend `--last-test-year`
past (current year - 5) without accepting that bias.

## 6. What to run each year (the annual checklist)

```bash
pip install -r requirements.txt

# 1. Refresh raw data (new class + updated Career AV for old classes)
python src/download_source_data.py
python src/build_consensus_board.py          # extend if source adds new years
python src/build_features.py --end-year <mature_year>

# 2. Rebuild the public board through the new class
python src/improve.py --feature-set profile --end-year <new_class_year>
python src/add_conservative_scores.py --factors "0.25,0.50,0.75"
python src/build_site.py

# 3. Re-validate on mature classes only (leave ~5 years for AV to mature)
python src/backtest.py --first-test-year 2011 --last-test-year <mature_year> --end-year <mature_year> --feature-set profile --apex-plus-factor 3.5
python src/predraft_backtest.py --first-test-year 2011 --last-test-year <mature_year> --end-year <mature_year>
python src/sweep_apex_factor.py --first-test-year 2011 --last-test-year <mature_year> --end-year <mature_year> --factors "0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5"
python src/position_models.py --first-test-year 2011 --last-test-year <mature_year> --end-year <mature_year> --feature-set profile --apex-plus-factor 3.5
python src/ablation_backtest.py --first-test-year 2011 --last-test-year <mature_year> --end-year <mature_year>

# 4. Run the gates before changing any public claim
python src/validation_gates.py reports/rolling_backtest_summary.csv --delta-col delta_raw_vs_pick_spearman_drafted --out reports/rolling_validation_gates.json
```

Or just trigger the GitHub Action (`Run APEX Backtests`), which runs all of the
above and deploys the dashboard.

When a new draft class's picks become available (late April), rerun steps 1-2.
When another class matures (~5 years of Career AV), bump `<mature_year>` by one
and rerun step 3 to re-verify the edge still holds.

## 7. Interpreting the board

- `apex_score` — headline score (conservative 0.50 blend of market and model).
- `apex_edge` — disagreement with the market; the most actionable output.
  Positive late-round edge = surplus candidate; negative early edge = bust risk.
- `implied_pick` / `pick_delta` — the same disagreement in draft-slot currency:
  "graded like pick #X, went #Y". Built by inverting the pick-to-outcome curve.
- `p_star`..`p_bust` — historical tier base rates for the player's grade
  bucket; the projection panel on each player card.
- Trust position-level edges more where the by-position backtest shows lift;
  be most skeptical of QB scores.

## 8. Failure modes to watch

1. **Censoring** — recent classes have incomplete Career AV; never validate on
   them and never celebrate lift measured on them.
2. **Leakage** — any new feature must be knowable before draft night. If it
   comes from a table that also has NFL outcomes, split carefully.
3. **Overfitting the sweep** — the APEX+ factor and shrinkage grids are tuned
   on validation folds; only trust settings that pass gates across many years.
4. **Source drift** — the upstream GitHub datasets can change schema; the
   downloader repairs malformed rows but check row counts year over year.
