from pathlib import Path
import json

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
# UX rebuild trigger: regenerate homepage with user-facing front-office labels.
DATA_PATH = ROOT / "data" / "apex_board.csv"
TEMPLATE_PATH = ROOT / "src" / "template.html"
TARGETS = [ROOT / "index.html", ROOT / "docs" / "index.html"]

SITE_COLS = [
    "Year",
    "Player",
    "Pos",
    "pos_g",
    "College",
    "Pick",
    "Rnd",
    "CarAV",
    "y",
    "apex_score",
    "apex_raw",
    "exp_at_pick",
    "apex_edge",
    "raw_edge",
    "apex_conservative_025",
    "apex_conservative_075",
    "model_status",
    "implied_pick",
    "pick_delta",
    "p_star",
    "p_starter",
    "p_contrib",
    "p_bust",
    "apex_pff",
    "pff_edge",
    "apex_live",
    "position_trust_label",
    "position_mean_delta",
    "position_win_rate",
    "position_worst_delta",
    "front_office_edge",
    "pick_bucket",
    "edge_band",
    "front_office_confidence",
    "front_office_call",
    "front_office_score",
    "front_office_status",
]

REQUIRED_INPUT_COLS = [
    "Year",
    "Player",
    "Pos",
    "pos_g",
    "Pick",
    "CarAV",
    "y",
    "apex",
    "exp_at_pick",
]


def first_existing(df: pd.DataFrame, candidates: list[str], fallback: float | str | None = None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return fallback


def load_board(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the generated board and fail loudly before publishing bad pages."""
    if not path.exists():
        raise FileNotFoundError(f"Generated board missing: {path}. Run src/improve.py before src/build_site.py.")
    if path.stat().st_size == 0:
        raise ValueError(f"Generated board is empty: {path}. Refusing to publish a zero-row dashboard.")
    try:
        board = pd.read_csv(path)
    except EmptyDataError as exc:
        raise ValueError(f"Generated board has no CSV rows: {path}. Refusing to publish a zero-row dashboard.") from exc
    if board.empty:
        raise ValueError(f"Generated board has zero rows: {path}. Refusing to publish a zero-row dashboard.")
    missing = [col for col in REQUIRED_INPUT_COLS if col not in board.columns]
    if missing:
        raise ValueError(f"Generated board is missing required columns {missing}: {path}")
    return board


df = load_board(DATA_PATH)
df["Pick"] = df["Pick"].where(df["Pick"] < 263)
df["College"] = first_existing(df, ["College", "college"], "Unknown")
df["College"] = df["College"].fillna("Unknown")

# Stage-3 "living projection": for classes with partial careers (1-3 NFL
# seasons of recorded AV), blend the pre-draft grade with the current
# within-class outcome percentile. Weight grows with seasons elapsed
# (heuristic, labeled as such on the site).
OUTCOME_DATA_YEAR = 2024  # last NFL season reflected in CarAV
seasons_elapsed = (OUTCOME_DATA_YEAR - df["Year"] + 1).clip(lower=0)
live_weight = (0.25 * seasons_elapsed).clip(upper=0.75)
partial = df["Year"].between(OUTCOME_DATA_YEAR - 2, OUTCOME_DATA_YEAR) & df["y"].notna()
df["apex_live"] = np.nan
if "apex_conservative_050" in df.columns:
    df.loc[partial, "apex_live"] = (
        (1 - live_weight[partial]) * pd.to_numeric(df.loc[partial, "apex_conservative_050"], errors="coerce")
        + live_weight[partial] * pd.to_numeric(df.loc[partial, "y"], errors="coerce")
    )

# Optional PFF-informed challenger scores (model outputs only; see
# src/build_pff_scores.py). Merged by Year+Player when the file exists.
PFF_SCORES_PATH = ROOT / "data" / "pff_scores.csv"
if PFF_SCORES_PATH.exists() and "apex_pff" not in df.columns:
    pff = pd.read_csv(PFF_SCORES_PATH)[["Year", "Player", "apex_pff", "pff_edge"]]
    df = df.merge(pff.drop_duplicates(["Year", "Player"]), on=["Year", "Player"], how="left")
    print(f"merged PFF-informed scores for {int(df['apex_pff'].notna().sum())} rows")

if "Rnd" not in df.columns:
    round_bins = [0, 32, 64, 100, 135, 176, 220, 262]
    df["Rnd"] = pd.cut(
        df["Pick"],
        bins=round_bins,
        labels=[1, 2, 3, 4, 5, 6, 7],
    ).astype("float")

# Site-facing score contract.
# Main score is the gate-passing APEX Conservative 0.50 candidate when present.
df["apex_raw"] = pd.to_numeric(df["apex"], errors="coerce")
df["apex_score"] = pd.to_numeric(first_existing(df, ["recommended_candidate_score", "apex_conservative_050", "apex"]), errors="coerce")
df["apex_edge"] = pd.to_numeric(first_existing(df, ["conservative_surplus_050", "surplus"], 0.0), errors="coerce")
df["raw_edge"] = pd.to_numeric(first_existing(df, ["surplus"], 0.0), errors="coerce")
if "apex_conservative_025" not in df.columns:
    df["apex_conservative_025"] = pd.to_numeric(df["exp_at_pick"], errors="coerce") + 0.25 * (df["apex_raw"] - pd.to_numeric(df["exp_at_pick"], errors="coerce"))
if "apex_conservative_075" not in df.columns:
    df["apex_conservative_075"] = pd.to_numeric(df["exp_at_pick"], errors="coerce") + 0.75 * (df["apex_raw"] - pd.to_numeric(df["exp_at_pick"], errors="coerce"))
if "model_status" not in df.columns:
    df["model_status"] = "apex_conservative_050_candidate"

# Guardrail / front-office labels added by src/apply_front_office_labels.py.
string_defaults = {
    "position_trust_label": "not_reviewed",
    "pick_bucket": "unknown",
    "edge_band": "neutral",
    "front_office_confidence": "low",
    "front_office_call": "hold_market",
    "front_office_status": "guardrail_only",
}
for col, default in string_defaults.items():
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)

numeric_defaults = {
    "position_mean_delta": np.nan,
    "position_win_rate": np.nan,
    "position_worst_delta": np.nan,
    "front_office_edge": df["apex_edge"],
    "front_office_score": df["apex_score"],
}
for col, default in numeric_defaults.items():
    if col not in df.columns:
        df[col] = default

for col in [
    "implied_pick",
    "pick_delta",
    "p_star",
    "p_starter",
    "p_contrib",
    "p_bust",
    "apex_pff",
    "pff_edge",
    "apex_live",
]:
    if col not in df.columns:
        df[col] = np.nan

for col in [
    "CarAV",
    "y",
    "apex_score",
    "apex_raw",
    "exp_at_pick",
    "apex_edge",
    "raw_edge",
    "apex_conservative_025",
    "apex_conservative_075",
    "implied_pick",
    "pick_delta",
    "p_star",
    "p_starter",
    "p_contrib",
    "p_bust",
    "apex_pff",
    "pff_edge",
    "apex_live",
    "position_mean_delta",
    "position_win_rate",
    "position_worst_delta",
    "front_office_edge",
    "front_office_score",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

data = df[SITE_COLS].copy()
rows = data.astype(object).where(pd.notnull(data), None).values.tolist()
if not rows:
    raise ValueError("Refusing to write dashboard with zero serialized rows.")
payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
html = TEMPLATE_PATH.read_text().replace("__DATA__", payload)

for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    out = html
    if target.parent.name == "docs":
        out = out.replace('href="docs/', 'href="')
    target.write_text(out)

print("rows:", len(rows), "size:", len(html) // 1024, "KB")
print("main_score: apex_conservative_050 via apex_score")
print("site_fields: front_office_call, confidence, position_trust_label")
