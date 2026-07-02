"""Score the board with the PFF-informed position-family challenger model.

Writes data/pff_scores.csv containing MODEL OUTPUTS ONLY (scores and edges),
which is safe to commit and publish. Raw PFF values never leave data/pff/ or
data/production/ (both gitignored).

The model is the position-family challenger: profile features plus whatever
per-position production features (ESPN QBR, PFF passing/receiving) clear the
recent-window coverage gate. Its validated lift is +0.0139 vs the market
(QBR config); PFF features are additional projection input for recent classes
and are NOT validated lift - the site labels this score experimental.

Run locally after refreshing PFF exports:
    python src/build_pff_scores.py
Then rebuild the site. In CI (no PFF data) the script still runs and scores
with whatever production features exist there (ESPN QBR only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import merge_optional_features
from feature_sets import model_features_for
from pipeline import ROOT, load_dataset, make_baseline, prepare_fold
from position_models import fit_position_residuals

OUT_PATH = ROOT / "data" / "pff_scores.csv"
DEFAULT_SHRINK = 0.4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    args = parser.parse_args()

    base_df = load_dataset(end_year=args.end_year)
    enriched, _ = merge_optional_features(base_df)
    train, scored = prepare_fold(enriched, enriched)

    feats = model_features_for("profile")
    base, _, _ = make_baseline(train)
    resid, fit_report = fit_position_residuals(train, base, feats=feats, min_train_rows=300)

    scored = scored.copy()
    market = base(scored)
    scored["apex_pff"] = np.clip(market + DEFAULT_SHRINK * resid(scored), 0.0, 1.0)
    scored["pff_edge"] = scored["apex_pff"] - market

    extras_used = {
        family: info.get("extra_features", [])
        for family, info in fit_report.get("families", {}).items()
        if info.get("trained")
    }
    out = scored[["Year", "Player", "pos_g", "apex_pff", "pff_edge"]].copy()
    out_path = Path(args.out)
    out.round(4).to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out)}")
    for family, extras in extras_used.items():
        print(f"  {family}: {len(extras)} production features active")


if __name__ == "__main__":
    main()
