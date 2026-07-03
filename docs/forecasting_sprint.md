# Zero-cost APEX forecasting improvement sprint

Goal: improve APEX as a forecasting product without paid APIs, paid data pulls, or unvalidated score promotion. The sprint treats the NFL draft market as the baseline to beat and only promotes changes that survive rolling, out-of-time validation.

## Operating rules

1. **No money spent.** Use the existing repo, public/free inputs already wired into the pipeline, local Python, and GitHub Actions.
2. **No headline promotion without gates.** A candidate must beat draft slot on mean lift, median lift, win rate, and worst-year behavior.
3. **Separate forecasting jobs.** Pre-draft ranking, post-draft market disagreement, live early-career updates, and experimental PFF/QB signals must stay visibly separate.
4. **Prefer labels before score changes.** If a signal is useful but not promotable, expose it as a trust label, flag, or research report rather than changing the public grade.

## Agent team

| Agent | Best efficient model / engine | Assignment | Deliverable |
|---|---|---|---|
| Orchestrator | GPT-5.5 Thinking + repo checks | Decide what is safe to implement and what must remain experimental. | Promotion/rejection decisions and repo changes. |
| Data QA Agent | pandas validation scripts | Prevent blank boards, leakage, stale generated files, and missing validation artifacts. | Build-blocking integrity checks. |
| Validation Agent | Existing LightGBM rolling backtests | Compare raw APEX, nested factor, position models, and ensembles against pick-only market baseline. | Year-by-year lift table and gate status. |
| Position Specialist Agent | Grouped out-of-time diagnostics | Identify position groups where disagreement should be trusted, shrunk, or scout-required. | Position trust table. |
| Calibration Agent | Historical bucket/base-rate calibration | Verify that value/fade bands actually map to career outperformance. | Edge calibration table and pick-bucket lift. |
| Product/Board Agent | Static-site contract checks | Keep the board honest and usable: separate default, challenger, experimental, and low-trust labels. | Dashboard/report guidance. |

## Implemented in this sprint

### 1. Forecasting audit runner

`src/forecasting_sprint.py` consumes existing validation artifacts and writes:

```bash
python src/forecasting_sprint.py --out-dir reports
```

Outputs:

```text
reports/forecasting_sprint_report.json
reports/forecasting_position_trust.csv
reports/forecasting_pick_bucket_lift.csv
reports/forecasting_edge_calibration.csv
reports/forecasting_agent_tasks.md
```

### 2. Front-office decision labels

`src/apply_front_office_labels.py` turns validated diagnostics into draft-room calls without promoting a new headline model:

```bash
python src/apply_front_office_labels.py \
  --reports-dir reports \
  --board data/apex_board.csv \
  --summary reports/front_office_board_report.json
```

It appends these board columns:

```text
position_trust_label
position_mean_delta
position_win_rate
position_worst_delta
pick_bucket
front_office_edge
edge_band
front_office_confidence
front_office_call
front_office_score
front_office_status
```

`front_office_call` can be `attack_value`, `value_watch`, `fade_watch`, `strong_fade`, `scout_required`, `model_note_only`, `hold_market`, or `prospect_watch`.

`front_office_score` is a guardrail score only. It shrinks the edge by position trust and is not the public headline unless a future backtest gate promotes it.

### 3. Site build guard

`src/build_site.py` refuses to publish a zero-row dashboard or a board missing required score/market columns.

## Forecasting roadmap

### Phase 1 — stop avoidable forecast/product failures

- Fail the build if `data/apex_board.csv` is empty.
- Fail the build if required score/market columns are missing.
- Generate a work queue after every validation run.
- Do not let an experimental model quietly become the public score.

### Phase 2 — improve reliability before raw lift

- Use position trust labels to mark where APEX has earned disagreement rights.
- Add pick-range trust labels for first round, Day 2, top 100, and Day 3.
- Gate severe negative windows before chasing higher average lift.
- Treat QB as scout-required unless its own validation improves.
- Treat any position with negative/wild rolling history as `model_note_only`, not an automatic attack/fade.

### Phase 3 — pre-draft forecasting research

- Keep ESPN/public consensus as the pre-draft market baseline.
- Test only pre-draft-safe features: combine/pro-day profile, age, college encoding, QB production, and consensus rank/grade.
- Never use actual pick, board-vs-pick, or `consensus_vs_pick` in a pre-draft model.
- Promote only if it beats consensus, not merely if it correlates with outcomes.

### Phase 4 — candidate model experiments

- Position-specific shrinkage and edge caps.
- Pick-bucket calibrated confidence labels.
- Nested conservative factor selection with stricter worst-year penalties.
- Separate bust-avoidance model for top-100 risk flags.
- PFF/QB challengers as labeled second opinions until they survive finished-career validation.

## Promotion gates

A scoring change can become public default only when it passes all four checks on the same rolling window:

| Gate | Required behavior |
|---|---|
| Mean lift | Positive vs pick-only baseline |
| Median lift | Positive vs pick-only baseline |
| Win rate | At least 8 of 11 years better than pick-only |
| Worst year | No catastrophic drawdown; target at least `-0.020` Spearman delta |

If a model fails any gate, it can still be useful, but only as an experimental report, confidence label, or scout-required warning.
