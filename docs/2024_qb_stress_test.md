# 2024 QB Stress Test: McCarthy vs. Maye/Daniels

## What went wrong

The old Prospect Lens QB production score used only four passing box-score
efficiency stats (INT rate, TD rate, YPA, final YPA). Result:

- J.J. McCarthy (efficient, low-volume, loaded Michigan roster): prod 0.844 -> `qb_model_greenlight`
- Bo Nix (high-completion quick-game scheme): prod 0.929 -> `qb_model_greenlight`
- Jayden Daniels (Heisman season built on elite rushing creation): prod 0.547 -> `qb_model_review`
- Drake Maye: prod 0.722 but lens score just under the 0.72 gate -> `qb_model_review`

Structural lesson: **an efficiency-only screen systematically rewards "clean"
low-volume profiles and punishes dual-threat creation**, which is exactly the
trait that drove Daniels' (and to a degree Maye's) early NFL value. A second
flaw compounded it: the site layer later overwrote McCarthy's label using his
partial NFL outcome — hindsight leakage disguised as calibration.

## What changed (2026-07)

1. **QB production is now two explicit sub-scores** in `build_prospect_lens.py`:
   - `qb_pass_efficiency_score` (INT rate, TD rate, YPA, final YPA)
   - `qb_creation_score` (rush YPC, final rush YPC, best total yards, total TD)
   - Combined 65% passing / 35% creation.
2. **Greenlight is harder and requires balance**: lens score >= 0.72, prod >= 0.55,
   pass >= 0.45 AND creation >= 0.45, zero caution flags, and an explicit
   efficiency-only exclusion. Efficiency alone can no longer clear the bar.
3. **New caution flag** `one_dimensional_efficiency_profile` fires when pass >= 0.70
   and creation <= 0.45 (the McCarthy shape; catches 12 QBs board-wide, all cautioned) and penalizes the lens score.
4. **No outcome leakage**: labels are pre-draft-information only. Early NFL
   evidence appears in `qb_lens_warning` / reasons as a separate badge and never
   rewrites the label.
5. **Reasons now explain the shape**: e.g. "efficiency-only profile;
   creation/volume unproven" vs "balanced passing + creation production".

## 2024 re-run (pre-draft info only)

| QB | Pass eff | Creation | Old call | New call |
|---|---|---|---|---|
| Caleb Williams | 0.50* | 0.26 | review | review (context needed; thin data) |
| Jayden Daniels | 0.55 | 0.45 | review | buy / volatile |
| Drake Maye | 0.72 | 0.67 | review | buy / volatile (most balanced) |
| J.J. McCarthy | 0.84 | 0.43 | **greenlight** | review (one_dimensional_efficiency_profile caution; site cannot upgrade cautioned QBs) |
| Bo Nix | 0.93 | 0.68 | **greenlight** | buy / volatile (creation 0.68 is real) |

A validation script (`tests/validate_qb_calibration.py`) now fails the build if
an efficiency-only shape is greenlit or uncautioned, if creation-driven QBs are
fade-penalized, if outcome fields re-enter label logic, or if the site layer
upgrades a model-cautioned QB. Verified against the generated 2024 output, not
hardcoded assumptions.

\* neutral due to missing CFBD coverage.

Maye and Daniels were **not** retroactively forced to green. The honest fix is
that no 2024 QB clears greenlight on pre-draft public data: the class was wide-
range, and the model now says so instead of manufacturing false certainty. The
McCarthy miss was a weighting error (creation absent), not random noise, and the
new gate structurally prevents that specific failure shape.

## Remaining limitations

- CFBD rush YPC includes sack yardage, deflating dual-threat creation slightly.
- No CPOE, pressure-to-sack, or charting (BTT/TWP) data in the free pipeline;
  `qb_scoring_system.py` supports those fields if data is ever supplied.
- Creation percentiles are board-wide across QBs with data; players with missing
  CFBD rows default to neutral 0.50.
