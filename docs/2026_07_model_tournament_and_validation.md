# 2026-07 Model Tournament and Validation

## QB calibration fix

The 2024 class exposed a QB production bias: an efficiency-only screen greenlit J.J. McCarthy/Bo Nix while leaving Jayden Daniels/Drake Maye in review. QB production is now split into passing-efficiency and rushing/creation sub-scores, greenlight requires a balanced profile, and early-NFL outcomes no longer rewrite pre-draft labels.

Full QB stress test: `docs/2024_qb_stress_test.md`.

## Coordinator verdict: no model promoted

A full challenger tournament (`src/model_tournament.py`), QB-only backtest (`src/qb_forecasting_backtest.py`), and calibration/position/miss reports (`src/agent_reports.py`) were added for mature classes (2011-2021). Verdicts:

- **No challenger promoted.** Stored mature-year scores are in-sample fits; the authoritative walk-forward evidence remains the rolling backtest: apex_raw about +0.014 mean lift vs draft slot, 9/11 win years. apex_plus is negative out-of-time and stays demoted.
- **No QB edge.** On drafted QBs, current model lift is small and unstable. Production-only scoring is strongly negative. The QB Lens stays explanatory-only and never overrides the main score.
- **Calibration warning:** p_starter/p_bust disagree heavily with simple y-threshold buckets; p_star is better calibrated. Tier odds are now labeled as model-defined buckets pending out-of-time recalibration.
- Position edge is directional only until true out-of-time retraining confirms it.

Generated reports include:

- `reports/model_tournament_report.json`
- `reports/qb_model_leaderboard.csv`
- `reports/calibration_report.json`
- `reports/qb_recent_forecast.csv`

## Public/default model

Unchanged: conservative/profile APEX remains the public/default model. The new work improves validation discipline, QB labeling, and dashboard honesty, not the promoted forecasting model.
