from pathlib import Path
import json

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

try:
    from calibrate_outcome_odds import add_display_odds
except Exception:
    add_display_odds = None

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "apex_board.csv"
TEMPLATE_PATH = ROOT / "src" / "template.html"
TARGETS = [ROOT / "index.html", ROOT / "docs" / "index.html"]
RECENT_START_YEAR = 2024
MAX_PICK = 262

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
    "projection_phase", "outcome_data_year", "qb_lens_label", "qb_lens_confidence",
    "qb_lens_warning", "qb_lens_reasons",
    "display_actual_pick", "display_model_pick", "display_slot_value", "display_slot_label",
    "display_star_pct", "display_starter_pct", "display_role_pct", "display_bust_pct",
    "display_bust_band", "odds_calibration_note",
]


def read_board(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing generated board: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except EmptyDataError as exc:
        raise ValueError(f"Board has no rows: {path}") from exc
    required = ["Year", "Player", "Pos", "pos_g", "Pick", "CarAV", "y", "apex", "exp_at_pick"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Board missing required columns: {missing}")
    return df


def numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def text_default(df: pd.DataFrame, col: str, default: str) -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)


def num_default(df: pd.DataFrame, col: str, default=np.nan) -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce")


def outcome_year(df: pd.DataFrame) -> int:
    years = numeric(df, "Year")[numeric(df, "y").notna()]
    return 2024 if years.dropna().empty else int(years.max())


def phase(df: pd.DataFrame) -> pd.Series:
    year = numeric(df, "Year")
    actual = numeric(df, "y")
    out = pd.Series("unknown", index=df.index, dtype="object")
    out[year.lt(RECENT_START_YEAR)] = "historical_validated"
    out[year.ge(RECENT_START_YEAR) & actual.isna()] = "projection_only"
    out[year.ge(RECENT_START_YEAR) & actual.notna()] = "projection_plus_partial_live"
    return out


def slot_label(slot_value: pd.Series) -> pd.Series:
    val = pd.to_numeric(slot_value, errors="coerce")
    out = pd.Series("No pick data", index=slot_value.index, dtype="object")
    out[val.notna() & val.between(-3, 3)] = "Fair value"
    out[val.gt(3)] = val[val.gt(3)].round().astype(int).map(lambda x: f"Value +{x} slots")
    out[val.lt(-3)] = val[val.lt(-3)].round().astype(int).map(lambda x: f"Reach {x} slots")
    return out


def bust_band(p: pd.Series) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce")
    out = pd.Series("Unknown", index=p.index, dtype="object")
    out[x.notna() & x.lt(0.16)] = "Low"
    out[x.notna() & x.ge(0.16) & x.lt(0.28)] = "Medium"
    out[x.notna() & x.ge(0.28)] = "High"
    return out


def fallback_display(df: pd.DataFrame) -> pd.DataFrame:
    pick = numeric(df, "Pick").where(lambda s: s.between(1, MAX_PICK))
    if "display_model_pick" not in df.columns:
        score = numeric(df, "apex_conservative_050", np.nan).fillna(numeric(df, "apex", 0))
        model_pick = pd.Series(np.nan, index=df.index, dtype="float64")
        rank_df = pd.DataFrame({"Year": numeric(df, "Year"), "score": score, "pick": pick, "Player": df["Player"].astype(str)}, index=df.index)
        for _, group in rank_df[rank_df["pick"].notna()].groupby("Year"):
            order = group.sort_values(["score", "pick", "Player"], ascending=[False, True, True]).index
            model_pick.loc[order] = np.arange(1, len(order) + 1, dtype=float)
        df["display_model_pick"] = model_pick.where(model_pick.between(1, MAX_PICK))
    if "display_actual_pick" not in df.columns:
        df["display_actual_pick"] = pick
    if "display_slot_value" not in df.columns:
        df["display_slot_value"] = numeric(df, "display_actual_pick") - numeric(df, "display_model_pick")
    if "display_slot_label" not in df.columns:
        df["display_slot_label"] = slot_label(df["display_slot_value"])
    for dst, src in [("display_star_pct", "p_star"), ("display_starter_pct", "p_starter"), ("display_role_pct", "p_contrib"), ("display_bust_pct", "p_bust")]:
        if dst not in df.columns:
            df[dst] = numeric(df, src)
    if "display_bust_band" not in df.columns:
        df["display_bust_band"] = bust_band(df["display_bust_pct"])
    if "odds_calibration_note" not in df.columns:
        df["odds_calibration_note"] = "Should-have-gone slot ranking; bust risk is historical slot miss risk, not exact certainty"
    return df


def clean_qb(df: pd.DataFrame) -> pd.DataFrame:
    qb = df.get("pos_g", pd.Series("", index=df.index)).astype(str).eq("QB") | df.get("Pos", pd.Series("", index=df.index)).astype(str).eq("QB")
    for col in ["qb_lens_label", "qb_lens_confidence", "qb_lens_warning", "qb_lens_reasons"]:
        text_default(df, col, "")
    label = df["qb_lens_label"].astype(str)
    needs = qb & label.isin(["", "nan", "None", "qb_model_greenlight", "qb_model_review"])
    score = numeric(df, "prospect_lens_score", 0.50).fillna(numeric(df, "apex_score", 0.50)).fillna(0.50)
    prod = numeric(df, "prospect_production_score", 0.50).fillna(0.50)
    edge = numeric(df, "front_office_edge", 0.0).fillna(numeric(df, "apex_edge", 0.0)).fillna(0.0)
    caution = numeric(df, "prospect_caution_count", 0).fillna(0)
    new = pd.Series("qb_review_context_needed", index=df.index, dtype="object")
    new[qb & (edge.le(-0.045) | prod.le(0.33) | score.lt(0.52))] = "qb_fade_risk"
    new[qb & score.ge(0.54) & score.lt(0.64) & edge.abs().le(0.035)] = "qb_market_aligned"
    new[qb & score.ge(0.64) & (prod.ge(0.50) | edge.ge(0.010)) & caution.eq(0)] = "qb_buy_volatile"
    new[qb & score.ge(0.76) & prod.ge(0.58) & caution.eq(0)] = "qb_buy_high_confidence"
    df.loc[needs, "qb_lens_label"] = new[needs]
    df.loc[qb, "prospect_lens_call"] = df.loc[qb, "qb_lens_label"]
    year = numeric(df, "Year")
    actual = numeric(df, "y")
    warn = pd.Series("", index=df.index, dtype="object")
    warn[qb & year.ge(RECENT_START_YEAR) & actual.isna()] = "projection_only_no_nfl_outcome"
    warn[qb & year.ge(RECENT_START_YEAR) & actual.notna() & actual.lt(0.45)] = "partial_live_below_projection_sample_warning"
    warn[qb & year.ge(RECENT_START_YEAR) & warn.eq("")] = "recent_class_partial_live_not_final"
    df.loc[qb & df["qb_lens_warning"].isin(["", "nan", "None"]), "qb_lens_warning"] = warn[qb]
    return df


df = read_board(DATA_PATH)
df["Pick"] = numeric(df, "Pick").where(lambda s: s < 263)
df["College"] = df["College"].fillna("Unknown") if "College" in df.columns else df.get("college", "Unknown")

outcome_data_year = outcome_year(df)
if "Rnd" not in df.columns:
    df["Rnd"] = pd.cut(df["Pick"], [0, 32, 64, 100, 135, 176, 220, 262], labels=[1, 2, 3, 4, 5, 6, 7]).astype("float")

df["apex_raw"] = numeric(df, "apex")
df["apex_score"] = numeric(df, "recommended_candidate_score", np.nan).fillna(numeric(df, "apex_conservative_050", np.nan)).fillna(numeric(df, "apex"))
df["apex_edge"] = numeric(df, "conservative_surplus_050", np.nan).fillna(numeric(df, "surplus", 0.0))
df["raw_edge"] = numeric(df, "surplus", 0.0)
if "apex_conservative_025" not in df.columns:
    df["apex_conservative_025"] = numeric(df, "exp_at_pick") + 0.25 * (df["apex_raw"] - numeric(df, "exp_at_pick"))
if "apex_conservative_075" not in df.columns:
    df["apex_conservative_075"] = numeric(df, "exp_at_pick") + 0.75 * (df["apex_raw"] - numeric(df, "exp_at_pick"))

text_defaults = {
    "model_status": "apex_conservative_050_candidate", "position_trust_label": "not_reviewed",
    "pick_bucket": "unknown", "edge_band": "neutral", "front_office_confidence": "low",
    "front_office_call": "hold_market", "front_office_status": "guardrail_only",
    "prospect_lens_call": "hold_grade", "prospect_lens_confidence": "low",
    "prospect_lens_status": "not_available", "prospect_production_signal": "profile_only",
    "prospect_caution_flags": "none", "projection_phase": "", "qb_lens_label": "",
    "qb_lens_confidence": "", "qb_lens_warning": "", "qb_lens_reasons": "",
}
for col, default in text_defaults.items():
    text_default(df, col, default)

if df["projection_phase"].isin(["", "nan", "None"]).any():
    df.loc[df["projection_phase"].isin(["", "nan", "None"]), "projection_phase"] = phase(df)
if "prospect_is_recent" not in df.columns:
    df["prospect_is_recent"] = numeric(df, "Year").ge(RECENT_START_YEAR) | numeric(df, "y").isna()

for col, default in {
    "position_mean_delta": np.nan, "position_win_rate": np.nan, "position_worst_delta": np.nan,
    "front_office_edge": df["apex_edge"], "front_office_score": df["apex_score"],
    "prospect_lens_score": df["apex_score"], "prospect_production_score": 0.50,
    "prospect_caution_count": 0, "prospect_signal_count": 0,
    "implied_pick": np.nan, "pick_delta": np.nan, "p_star": np.nan,
    "p_starter": np.nan, "p_contrib": np.nan, "p_bust": np.nan,
    "apex_pff": np.nan, "pff_edge": np.nan, "apex_live": np.nan,
}.items():
    if col not in df.columns:
        df[col] = default

df["outcome_data_year"] = outcome_data_year
if add_display_odds is not None:
    try:
        df = add_display_odds(df)
    except Exception as exc:
        print(f"display odds calibration failed; using fallback display fields: {exc}")
df = fallback_display(df)
df = clean_qb(df)

numeric_cols = [c for c in SITE_COLS if c not in {"Player", "Pos", "pos_g", "College", "model_status", "position_trust_label", "pick_bucket", "edge_band", "front_office_confidence", "front_office_call", "front_office_status", "prospect_lens_call", "prospect_lens_confidence", "prospect_lens_status", "prospect_production_signal", "prospect_caution_flags", "projection_phase", "qb_lens_label", "qb_lens_confidence", "qb_lens_warning", "qb_lens_reasons", "display_slot_label", "display_bust_band", "odds_calibration_note"}]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
for col in SITE_COLS:
    if col not in df.columns:
        df[col] = None

df.round(4).to_csv(DATA_PATH, index=False)
rows = df[SITE_COLS].astype(object).where(pd.notnull(df[SITE_COLS]), None).values.tolist()
if not rows:
    raise ValueError("Refusing to write dashboard with zero serialized rows.")
html = TEMPLATE_PATH.read_text().replace("__DATA__", json.dumps(rows, separators=(",", ":"), allow_nan=False))
if "__DATA__" in html:
    raise ValueError("Template data placeholder was not replaced.")
for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    out = html.replace('href="docs/', 'href="') if target.parent.name == "docs" else html
    target.write_text(out)
print("rows:", len(rows), "outcome_data_year:", outcome_data_year)
print("site_fields: should-have-gone display_model_pick is unique by year")
