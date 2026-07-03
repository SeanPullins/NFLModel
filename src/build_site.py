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
    "projection_phase", "outcome_data_year", "qb_lens_label", "qb_lens_confidence",
    "qb_lens_warning", "qb_lens_reasons",
    "display_actual_pick", "display_model_pick", "display_slot_value", "display_slot_label",
    "display_star_pct", "display_starter_pct", "display_role_pct", "display_bust_pct",
    "display_bust_band", "odds_calibration_note",
]

REQUIRED_INPUT_COLS = ["Year", "Player", "Pos", "pos_g", "Pick", "CarAV", "y", "apex", "exp_at_pick"]
RECENT_START_YEAR = 2024
MAX_PICK = 262


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


def detect_outcome_data_year(board: pd.DataFrame) -> int:
    years = pd.to_numeric(board.loc[pd.to_numeric(board["y"], errors="coerce").notna(), "Year"], errors="coerce").dropna()
    return 2024 if years.empty else int(years.max())


def default_projection_phase(board: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(board["Year"], errors="coerce")
    actual = pd.to_numeric(board["y"], errors="coerce")
    out = pd.Series("unknown", index=board.index, dtype="object")
    out[year.lt(RECENT_START_YEAR)] = "historical_validated"
    out[year.ge(RECENT_START_YEAR) & actual.isna()] = "projection_only"
    out[year.ge(RECENT_START_YEAR) & actual.notna()] = "projection_plus_partial_live"
    return out


def make_slot_label(slot_value: pd.Series) -> pd.Series:
    val = pd.to_numeric(slot_value, errors="coerce")
    out = pd.Series("No pick data", index=slot_value.index, dtype="object")
    out[val.notna() & val.between(-3, 3)] = "Fair value"
    out[val.gt(3)] = val[val.gt(3)].round().astype(int).map(lambda x: f"Value +{x} slots")
    out[val.lt(-3)] = val[val.lt(-3)].round().astype(int).map(lambda x: f"Reach {x} slots")
    return out


def make_bust_band(pct: pd.Series) -> pd.Series:
    p = pd.to_numeric(pct, errors="coerce")
    out = pd.Series("Unknown", index=pct.index, dtype="object")
    out[p.notna() & p.lt(0.18)] = "Low"
    out[p.notna() & p.ge(0.18) & p.lt(0.32)] = "Medium"
    out[p.notna() & p.ge(0.32) & p.lt(0.48)] = "High"
    out[p.notna() & p.ge(0.48)] = "Very High"
    return out


def add_display_defaults(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    pick = pd.to_numeric(out.get("Pick", pd.Series(np.nan, index=out.index)), errors="coerce").where(lambda x: x.between(1, MAX_PICK))
    model_pick = pd.to_numeric(out.get("implied_pick", pd.Series(np.nan, index=out.index)), errors="coerce").where(lambda x: x.between(1, MAX_PICK))
    slot_value = pick - model_pick

    if "display_actual_pick" not in out.columns:
        out["display_actual_pick"] = pick
    if "display_model_pick" not in out.columns:
        out["display_model_pick"] = model_pick
    if "display_slot_value" not in out.columns:
        out["display_slot_value"] = slot_value
    if "display_slot_label" not in out.columns:
        out["display_slot_label"] = make_slot_label(out["display_slot_value"])
    if "display_star_pct" not in out.columns:
        out["display_star_pct"] = pd.to_numeric(out.get("p_star", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "display_starter_pct" not in out.columns:
        out["display_starter_pct"] = pd.to_numeric(out.get("p_starter", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "display_role_pct" not in out.columns:
        out["display_role_pct"] = pd.to_numeric(out.get("p_contrib", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "display_bust_pct" not in out.columns:
        out["display_bust_pct"] = pd.to_numeric(out.get("p_bust", pd.Series(np.nan, index=out.index)), errors="coerce")
    if "display_bust_band" not in out.columns:
        out["display_bust_band"] = make_bust_band(out["display_bust_pct"])
    if "odds_calibration_note" not in out.columns:
        out["odds_calibration_note"] = "Model-defined outcome buckets; run calibrate_outcome_odds.py for slot-calibrated display odds"
    return out


def clean_qb_labels(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    qb = out.get("pos_g", pd.Series("", index=out.index)).astype(str).eq("QB") | out.get("Pos", pd.Series("", index=out.index)).astype(str).eq("QB")
    for col in ["qb_lens_label", "qb_lens_confidence", "qb_lens_warning", "qb_lens_reasons"]:
        if col not in out.columns:
            out[col] = ""

    actual = pd.to_numeric(out.get("y", pd.Series(np.nan, index=out.index)), errors="coerce")
    score = pd.to_numeric(out.get("prospect_lens_score", out.get("apex_score", pd.Series(0.50, index=out.index))), errors="coerce").fillna(0.50)
    prod = pd.to_numeric(out.get("prospect_production_score", pd.Series(0.50, index=out.index)), errors="coerce").fillna(0.50)
    edge = pd.to_numeric(out.get("front_office_edge", out.get("apex_edge", pd.Series(0.0, index=out.index))), errors="coerce").fillna(0.0)
    year = pd.to_numeric(out.get("Year", pd.Series(np.nan, index=out.index)), errors="coerce")

    warn = pd.Series("", index=out.index, dtype="object")
    warn[qb & year.ge(RECENT_START_YEAR) & actual.isna()] = "projection_only_no_nfl_outcome"
    warn[qb & year.ge(RECENT_START_YEAR) & actual.notna() & actual.lt(0.45)] = "partial_live_below_projection_sample_warning"
    warn[qb & year.ge(RECENT_START_YEAR) & warn.eq("")] = "recent_class_partial_live_not_final"
    out.loc[qb & out["qb_lens_warning"].astype(str).isin(["", "nan", "None"]), "qb_lens_warning"] = warn[qb]

    label = out["qb_lens_label"].astype(str)
    needs = qb & label.isin(["", "nan", "None", "qb_model_greenlight", "qb_model_review"])
    new_label = pd.Series("qb_review_context_needed", index=out.index, dtype="object")
    # Labels are pre-draft-information only. Early NFL outcomes never change
    # the label; they surface through the warning/evidence badge.
    qb_pass = pd.to_numeric(out.get("qb_pass_efficiency_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    qb_create = pd.to_numeric(out.get("qb_creation_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    balanced = qb_pass.ge(0.45) & qb_create.ge(0.45)
    caution_n = pd.to_numeric(out.get("prospect_caution_count", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    new_label[qb & (edge.le(-0.045) | prod.le(0.33) | score.lt(0.52))] = "qb_fade_risk"
    new_label[qb & score.ge(0.54) & score.lt(0.64) & edge.abs().le(0.035)] = "qb_market_aligned"
    new_label[qb & score.ge(0.64) & (prod.ge(0.50) | edge.ge(0.010)) & caution_n.eq(0)] = "qb_buy_volatile"
    new_label[qb & score.ge(0.76) & prod.ge(0.58) & balanced & caution_n.eq(0)] = "qb_buy_high_confidence"
    out.loc[needs, "qb_lens_label"] = new_label[needs]
    out.loc[qb, "prospect_lens_call"] = out.loc[qb, "qb_lens_label"]

    conf = out["qb_lens_confidence"].astype(str)
    conf_needs = qb & conf.isin(["", "nan", "None"])
    out.loc[conf_needs, "qb_lens_confidence"] = out.loc[conf_needs, "prospect_lens_confidence"].fillna("medium")
    out.loc[qb, "prospect_lens_confidence"] = out.loc[qb, "qb_lens_confidence"]

    reasons = out["qb_lens_reasons"].astype(str)
    reasons_need = qb & reasons.isin(["", "nan", "None"])
    fair = pd.to_numeric(out.get("display_model_pick", out.get("implied_pick", pd.Series(np.nan, index=out.index))), errors="coerce")
    pick = pd.to_numeric(out.get("display_actual_pick", out.get("Pick", pd.Series(np.nan, index=out.index))), errors="coerce")
    generated_reasons = []
    for i in out.index:
        if not reasons_need.loc[i]:
            generated_reasons.append(out.at[i, "qb_lens_reasons"])
            continue
        parts = []
        if pd.notna(fair.loc[i]) and pd.notna(pick.loc[i]):
            parts.append(f"model pick #{int(round(fair.loc[i]))} vs actual #{int(round(pick.loc[i]))}")
        else:
            parts.append("profile/market projection")
        p, c = qb_pass.loc[i], qb_create.loc[i]
        if pd.notna(p) and pd.notna(c):
            if p >= 0.70 and c <= 0.45:
                parts.append("efficient passer; creation/volume unproven")
            elif c >= 0.70 and p <= 0.35:
                parts.append("creation-driven profile; passing efficiency lags")
            elif p >= 0.55 and c >= 0.55:
                parts.append("balanced passing + creation production")
            else:
                parts.append(f"pass efficiency {p:.0%} / creation {c:.0%}")
        elif prod.loc[i] >= 0.58:
            parts.append("production layer supports")
        elif prod.loc[i] <= 0.42:
            parts.append("production caution")
        else:
            parts.append("production mixed/incomplete")
        if pd.notna(actual.loc[i]):
            parts.append("partial NFL evidence confirming" if actual.loc[i] >= 0.55 else "partial NFL evidence below projection")
        else:
            parts.append("projection only")
        generated_reasons.append(" | ".join(parts[:3]))
    out.loc[reasons_need, "qb_lens_reasons"] = pd.Series(generated_reasons, index=out.index)[reasons_need]
    return out


df = load_board(DATA_PATH)
df["Pick"] = df["Pick"].where(df["Pick"] < 263)
df["College"] = first_existing(df, ["College", "college"], "Unknown")
df["College"] = df["College"].fillna("Unknown")

OUTCOME_DATA_YEAR = detect_outcome_data_year(df)
seasons_elapsed = (OUTCOME_DATA_YEAR - pd.to_numeric(df["Year"], errors="coerce") + 1).clip(lower=0)
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
    "projection_phase": "",
    "qb_lens_label": "",
    "qb_lens_confidence": "",
    "qb_lens_warning": "",
    "qb_lens_reasons": "",
    "display_slot_label": "No pick data",
    "display_bust_band": "Unknown",
    "odds_calibration_note": "Slot-calibrated display odds not yet run",
}
for col, default in string_defaults.items():
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str)

df.loc[df["projection_phase"].isin(["", "nan", "None"]), "projection_phase"] = default_projection_phase(df)

if "prospect_is_recent" not in df.columns:
    df["prospect_is_recent"] = pd.to_numeric(df["Year"], errors="coerce").ge(RECENT_START_YEAR) | df["y"].isna()
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

df["outcome_data_year"] = OUTCOME_DATA_YEAR
df = add_display_defaults(df)
df = clean_qb_labels(df)

numeric_cols = [
    "CarAV", "y", "apex_score", "apex_raw", "exp_at_pick", "apex_edge", "raw_edge",
    "apex_conservative_025", "apex_conservative_075", "implied_pick", "pick_delta",
    "p_star", "p_starter", "p_contrib", "p_bust", "apex_pff", "pff_edge", "apex_live",
    "position_mean_delta", "position_win_rate", "position_worst_delta", "front_office_edge",
    "front_office_score", "prospect_lens_score", "prospect_production_score",
    "prospect_caution_count", "prospect_signal_count", "outcome_data_year",
    "display_actual_pick", "display_model_pick", "display_slot_value",
    "display_star_pct", "display_starter_pct", "display_role_pct", "display_bust_pct",
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

print("rows:", len(rows), "size:", len(html) // 1024, "KB", "outcome_data_year:", OUTCOME_DATA_YEAR)
print("site_fields: display_model_pick, display_slot_value, display_bust_pct, qb_lens_label")
