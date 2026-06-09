# APEX Draft Model

Two-stage residual NFL draft model + interactive scouting dashboard covering historical classes and forward-looking prospect watchlists. Out-of-time holdout (2012–14 classes): **Spearman ρ 0.630** vs 0.619 pick-only market baseline.

## Live dashboard
Deploy free on GitHub Pages:
```bash
git init && git add . && git commit -m "APEX v1.1"
gh repo create apex-draft-model --public --source=. --push
```
Then: repo **Settings → Pages → Source: main / `/docs`**. Dashboard goes live at `https://<you>.github.io/apex-draft-model/` — searchable 10,398-player board from 2000–2026, prospect filters, surplus-value lenses, and pick-vs-outcome scatter with the market curve drawn in.

## Architecture
1. **Market baseline** — isotonic regressions pick→outcome, blended global + per-position curves
2. **Athletic residual** — 5-seed bagged LightGBM (15 leaves, λ₂=5) on position-normalized combine z-scores (speed score, explosion, agility, BMI), age, shrunken college encoding — predicts deviation from market price
3. **Per-position shrinkage** — residual weight tuned per position on 2010–11 validation (OL 0.6, WR 0.7, EDGE/LB 0.0)

**Target:** within-class Career AV percentile — immune to career-length censoring.
**Key finding:** naive feature-stacking *loses* to draft capital (ρ 0.56 vs 0.62); only the residual design extracts net signal, concentrated at OL (0.674 vs 0.651) and EDGE.

## Repo layout
```
src/        pipeline.py (features/data) · improve.py (train+eval) · build_site.py + template.html (dashboard)
models/     5 bagged LightGBM boosters + isotonic baselines/shrinkage (joblib)
data/       apex_board.csv (expanded 2000–2026 board) · SOURCES.md
docs/       index.html — self-contained dashboard (GitHub Pages root)
```
Retrain: `pip install -r requirements.txt && python src/improve.py`

## Roadmap
- v2: stack college production scores (PIPE+ / PFF) with `talent_resid` — production is the public market's largest documented mispricing
- Extend outcome validation as newer classes accumulate meaningful Career AV
