"""Validate display outcome odds and exact slot fields.

Run after:
  python src/calibrate_outcome_odds.py
  python src/build_site.py
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []
board = pd.read_csv(ROOT / "data" / "apex_board.csv", low_memory=False)

required = [
    "display_actual_pick", "display_model_pick", "display_slot_value", "display_slot_label",
    "display_star_pct", "display_starter_pct", "display_role_pct", "display_bust_pct",
    "display_bust_band", "odds_calibration_note",
]
missing = [c for c in required if c not in board.columns]
if missing:
    errors.append(f"missing display columns: {missing}")

if not missing:
    for col in ["display_star_pct", "display_starter_pct", "display_role_pct", "display_bust_pct"]:
        values = pd.to_numeric(board[col], errors="coerce")
        bad = values.notna() & ~values.between(0, 1)
        if bad.any():
            errors.append(f"{col} has values outside 0-1")

    drafted = board[pd.to_numeric(board["Pick"], errors="coerce").between(1, 262)].copy()
    bust = pd.to_numeric(drafted["display_bust_pct"], errors="coerce").dropna()
    if len(bust) >= 50:
        if bust.std() < 0.025:
            errors.append(f"display_bust_pct is too compressed; std={bust.std():.4f}")
        if bust.nunique() < 15:
            errors.append(f"display_bust_pct has too few unique values: {bust.nunique()}")
    bands = set(drafted["display_bust_band"].dropna().astype(str))
    if len(bands - {"Unknown"}) < 3:
        errors.append(f"not enough bust-risk bands represented: {sorted(bands)}")

    implied = pd.to_numeric(board.get("implied_pick"), errors="coerce")
    model = pd.to_numeric(board.get("display_model_pick"), errors="coerce")
    if implied.notna().any() and model[implied.notna()].isna().mean() > 0.05:
        errors.append("display_model_pick is missing for too many rows with implied_pick")

    labels = board["display_slot_label"].dropna().astype(str)
    if labels.str.contains("round", case=False, regex=False).any():
        errors.append("display_slot_label uses broad round language instead of exact slots")
    if not labels.str.contains("Value|Reach|Fair value|No pick data", regex=True).any():
        errors.append("display_slot_label does not contain exact slot language")

if errors:
    print("FAIL")
    for error in errors:
        print(" -", error)
    sys.exit(1)
print("PASS: display outcome odds and exact slot fields validated")
