from pathlib import Path
import json

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "apex_board.csv"
TEMPLATE_PATH = ROOT / "src" / "template.html"
TARGETS = [ROOT / "index.html", ROOT / "docs" / "index.html"]

SITE_COLS = [
    "Year", "Player", "Pos", "pos_g", "College", "Pick", "Rnd", "CarAV", "y",
    "apex_score", "apex_raw", "exp_at_pick", "apex_edge", "raw_edge",
    "apex_conservative_025", "apex_conservative_075", "model_status", "implied_pick", "pick_delta",
    "p_star", "p_starter", "p_contrib", "p_bust", "apex_pff", "pff_edge", "apex_live",
    "position_trust_label", "position_mean_delta", "position_win_rate", "position_worst_delta",
    "front_office_edge", "pick_bucket", "edge_band", "front_office_confidence", "front_office_call",
    "front_office_score", "front_office_status",
    "prospect_is_recent", "prospect_lens_score", "prospect_lens_call", "prospect_lens_confidence",
    "prospect_lens_status", "prospect_production_score", "prospect_production_signal",
    "prospect_caution_flags", "prospect_caution_count", "prospect_signal_count",
]

REQUIRED_INPUT_COLS = ["Year", "Player", "Pos", "pos_g", "Pick", "CarAV", "y", "apex", "exp_at_pick"]


def first_existing(df: pd.DataFrame, candidates: list[str], fallback: float | str | None = None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return fallback


def load_board(path: Path = DATA_PATH) -> pd.DataFrame:
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

OUTCOME_DATA_YEAR = 2024
seasons_elapsed = (pd.to_numeric(df["Year"], errors="coerce") - 0)
seasons_elapsed = (OUTCOME_DATA_YEAR - seasons_elapsed + 1).clip(lower=0)
live_weight = (0.25 * seasons_elapsed).clip(upper=0.75)
partial = df["Year"].between(OUTCOME_DATA_YEAR - 2, OUTCOME_DATA_YEAR) & df["y"].notna()
df["apex_live"] = np.nan
if "apex_conservative_050" in df.columns:
    df.loc[partial, "apex_live"] = (
        (1 - live_weight[partial]) * pd.to_numeric(df.loc[partial, "apex_conservative_050"], errors="coerce")
        + live_weight[partial] * pd.to_numeric(df.loc[partial, "y"], errors="coerce")
    )

PFF_SCORES_PATH = ROOT / "data" / "pff_scores.csv"
if PFF_SCORES_PATH.exists() and "apex_pff" not in df.columns:
    pff = pd.read_csv(PFF_SCORES_PATH)[["Year", "Player", "apex_pff", "pff_edge"]]
    df = df.merge(pff.drop_duplicates(["Year", "Player"]), on=["Year", "Player"], how="left")
    print(f"merged PFF-informed scores for {int(df['apex_pff'].notna().sum())} rows")

if "Rnd" not in df.columns:
    round_bins = [0, 32, 64, 100, 135, 176, 220, 262]
    df["Rnd"] = pd.cut(df["Pick"], bins=round_bins, labels=[1, 2, 3, 4, 5, 6, 7]).astype("float")

# Site-facing score contract.
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

string_defaults = {
    "position_trust_label": "not_reviewed",
    "pick_bucket": "unknown",
    "edge_band": "neutral",
    "front_office_confidence": "low",
    "front_office_call": "hold_market",
    "front_office_status": "guardrail_only",
    "prospect_lens_call": "hold_grade",
    "prospect_lens_confidence": "low",
    "prospect_lens_status": "not_available",
    "prospect_production_signal": "profile_only",
    "prospect_caution_flags": "none",
}
for col, default in string_defaults.items():
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)

if "prospect_is_recent" not in df.columns:
    df["prospect_is_recent"] = pd.to_numeric(df["Year"], errors="coerce").ge(2024) | df["y"].isna()
else:
    df["prospect_is_recent"] = df["prospect_is_recent"].fillna(False).astype(bool)

numeric_defaults = {
    "position_mean_delta": np.nan,
    "position_win_rate": np.nan,
    "position_worst_delta": np.nan,
    "front_office_edge": df["apex_edge"],
    "front_office_score": df["apex_score"],
    "prospect_lens_score": first_existing(df, ["front_office_score", "apex_score"], 0.50),
    "prospect_production_score": 0.50,
    "prospect_caution_count": 0,
    "prospect_signal_count": 0,
}
for col, default in numeric_defaults.items():
    if col not in df.columns:
        df[col] = default

for col in ["implied_pick", "pick_delta", "p_star", "p_starter", "p_contrib", "p_bust", "apex_pff", "pff_edge", "apex_live"]:
    if col not in df.columns:
        df[col] = np.nan

numeric_cols = [
    "CarAV", "y", "apex_score", "apex_raw", "exp_at_pick", "apex_edge", "raw_edge",
    "apex_conservative_025", "apex_conservative_075", "implied_pick", "pick_delta",
    "p_star", "p_starter", "p_contrib", "p_bust", "apex_pff", "pff_edge", "apex_live",
    "position_mean_delta", "position_win_rate", "position_worst_delta", "front_office_edge",
    "front_office_score", "prospect_lens_score", "prospect_production_score",
    "prospect_caution_count", "prospect_signal_count",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

data = df[SITE_COLS].copy()
rows = data.astype(object).where(pd.notnull(data), None).values.tolist()
if not rows:
    raise ValueError("Refusing to write dashboard with zero serialized rows.")
payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
html = TEMPLATE_PATH.read_text().replace("__DATA__", payload)
if "__DATA__" in html:
    raise ValueError("Template data placeholder was not replaced; refusing to publish blank dashboard.")

for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    out = html
    if target.parent.name == "docs":
        out = out.replace('href="docs/', 'href="')
    target.write_text(out)

print("rows:", len(rows), "size:", len(html) // 1024, "KB")
print("site_fields: prospect_lens_call, prospect_lens_score, front_office_call")
