"""Apply validated front-office decision labels to the public APEX board.

This does not promote a new model score. It uses out-of-time validation reports to
add decision hygiene: attack, fade, watch, or force a scout override.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCOUT_REQUIRED_POSITIONS = {"QB"}
TRUST_MULTIPLIER = {
    "candidate_for_more_trust": 0.50,
    "track_but_do_not_promote": 0.40,
    "neutral_watch": 0.25,
    "insufficient_sample": 0.20,
    "shrink_or_suppress": 0.10,
    "scout_required": 0.15,
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def one_num(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def pick_bucket(pick: Any) -> str:
    p = one_num(pick)
    if pd.isna(p):
        return "undrafted_or_prospect"
    if p <= 32:
        return "round_1"
    if p <= 64:
        return "round_2"
    if p <= 100:
        return "top_100_day_2"
    if p <= 150:
        return "early_day_3"
    if p <= 262:
        return "late_day_3"
    return "udfa"


def edge_band(edge: Any) -> str:
    e = one_num(edge)
    if pd.isna(e):
        return "no_edge"
    if e >= 0.075:
        return "strong_value"
    if e >= 0.035:
        return "value"
    if e <= -0.075:
        return "strong_fade"
    if e <= -0.035:
        return "fade"
    return "neutral"


def infer_position_trust(reports_dir: Path) -> pd.DataFrame:
    existing = read_csv(reports_dir / "forecasting_position_trust.csv")
    if not existing.empty and {"pos_g", "trust_label"}.issubset(existing.columns):
        keep = [c for c in ["pos_g", "trust_label", "mean_delta", "win_rate", "worst_delta"] if c in existing.columns]
        return existing[keep].drop_duplicates("pos_g")

    pos = read_csv(reports_dir / "rolling_backtest_by_position.csv")
    if pos.empty or "delta_raw_vs_pick" not in pos.columns:
        return pd.DataFrame(columns=["pos_g", "trust_label", "mean_delta", "win_rate", "worst_delta"])

    rows = []
    for group, frame in pos.groupby("pos_g"):
        d = pd.to_numeric(frame["delta_raw_vs_pick"], errors="coerce").dropna()
        if d.empty:
            continue
        mean, win_rate, worst = float(d.mean()), float((d > 0).mean()), float(d.min())
        if mean > 0.02 and worst > -0.02 and win_rate >= 0.70:
            label = "candidate_for_more_trust"
        elif mean > 0 and win_rate >= 0.60:
            label = "track_but_do_not_promote"
        elif mean < 0 or worst < -0.05:
            label = "shrink_or_suppress"
        else:
            label = "neutral_watch"
        rows.append({
            "pos_g": str(group),
            "trust_label": label,
            "mean_delta": round(mean, 4),
            "win_rate": round(win_rate, 4),
            "worst_delta": round(worst, 4),
        })
    return pd.DataFrame(rows)


def confidence(row: pd.Series) -> str:
    trust = str(row.get("position_trust_label", "insufficient_sample"))
    band = str(row.get("edge_band", "neutral"))
    if trust == "scout_required":
        return "scout_required"
    if trust == "candidate_for_more_trust" and band in {"strong_value", "strong_fade"}:
        return "high"
    if trust in {"candidate_for_more_trust", "track_but_do_not_promote"} and band != "neutral":
        return "medium"
    if trust == "shrink_or_suppress" and band != "neutral":
        return "low_model_only"
    return "low"


def front_office_call(row: pd.Series) -> str:
    trust = str(row.get("position_trust_label", "insufficient_sample"))
    band = str(row.get("edge_band", "neutral"))
    pick = one_num(row.get("Pick"))
    if pd.isna(pick):
        return "prospect_watch"
    if trust == "scout_required":
        return "scout_required"
    if trust == "shrink_or_suppress" and band != "neutral":
        return "model_note_only"
    if band == "strong_value" and trust in {"candidate_for_more_trust", "track_but_do_not_promote"}:
        return "attack_value"
    if band == "value" and trust in {"candidate_for_more_trust", "track_but_do_not_promote", "neutral_watch"}:
        return "value_watch"
    if band == "strong_fade" and trust in {"candidate_for_more_trust", "track_but_do_not_promote", "neutral_watch"}:
        return "strong_fade"
    if band == "fade" and trust in {"candidate_for_more_trust", "track_but_do_not_promote", "neutral_watch"}:
        return "fade_watch"
    return "hold_market"


def apply_labels(board: pd.DataFrame, trust: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        raise ValueError("Board is empty; cannot apply front-office labels.")
    if "pos_g" not in board.columns or "exp_at_pick" not in board.columns:
        raise ValueError("Board must include pos_g and exp_at_pick.")
    out = board.copy()
    if trust.empty:
        out["position_trust_label"] = "insufficient_sample"
        out["position_mean_delta"] = np.nan
        out["position_win_rate"] = np.nan
        out["position_worst_delta"] = np.nan
    else:
        trust = trust.rename(columns={
            "trust_label": "position_trust_label",
            "mean_delta": "position_mean_delta",
            "win_rate": "position_win_rate",
            "worst_delta": "position_worst_delta",
        })
        out = out.merge(trust, on="pos_g", how="left")
        out["position_trust_label"] = out["position_trust_label"].fillna("insufficient_sample")

    out.loc[out["pos_g"].astype(str).isin(SCOUT_REQUIRED_POSITIONS), "position_trust_label"] = "scout_required"
    edge_source = "conservative_surplus_050" if "conservative_surplus_050" in out.columns else "surplus"
    out["front_office_edge"] = pd.to_numeric(out[edge_source], errors="coerce")
    out["pick_bucket"] = out["Pick"].map(pick_bucket) if "Pick" in out.columns else "unknown"
    out["edge_band"] = out["front_office_edge"].map(edge_band)
    out["front_office_confidence"] = out.apply(confidence, axis=1)
    out["front_office_call"] = out.apply(front_office_call, axis=1)
    multipliers = out["position_trust_label"].map(TRUST_MULTIPLIER).fillna(0.20)
    market = pd.to_numeric(out["exp_at_pick"], errors="coerce")
    out["front_office_score"] = (market + multipliers * out["front_office_edge"]).clip(0.01, 0.99)
    out["front_office_status"] = "guardrail_not_headline_score"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default=str(ROOT / "data" / "apex_board.csv"))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--out", default=None, help="Defaults to overwriting --board")
    parser.add_argument("--summary", default=str(ROOT / "reports" / "front_office_board_report.json"))
    args = parser.parse_args()

    board_path = Path(args.board)
    out_path = Path(args.out) if args.out else board_path
    reports_dir = Path(args.reports_dir)
    labeled = apply_labels(read_csv(board_path), infer_position_trust(reports_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.round(4).to_csv(out_path, index=False)
    summary = {
        "rows": int(len(labeled)),
        "calls": labeled["front_office_call"].value_counts(dropna=False).to_dict(),
        "confidence": labeled["front_office_confidence"].value_counts(dropna=False).to_dict(),
        "position_trust": labeled["position_trust_label"].value_counts(dropna=False).to_dict(),
        "score_contract": "front_office_score is a guardrail score only; public headline remains validated APEX 0.50 unless future gates promote it.",
        "uses_paid_data_or_apis": False,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
