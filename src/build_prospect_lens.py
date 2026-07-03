"""Build a newer-class Prospect Lens layer for the public board.

The goal is not to replace the validated APEX score. It gives recent classes and
future prospects a real model-facing read instead of collapsing to Scout Required.

Inputs are free/internal pipeline fields only:
- conservative APEX / expected value at pick
- front-office edge and position trust labels
- draft slot / pick bucket
- optional CFBD production features when the repo secret has produced them

Outputs are appended to data/apex_board.csv and summarized in reports/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "data" / "apex_board.csv"
CFBD_PATH = ROOT / "data" / "production" / "cfbd_production.csv"
REPORT_PATH = ROOT / "reports" / "prospect_lens_report.json"
RECENT_NOTES_PATH = ROOT / "reports" / "prospect_lens_recent.csv"

RECENT_START_YEAR = 2024

TRUST_BONUS = {
    "candidate_for_more_trust": 0.060,
    "track_but_do_not_promote": 0.040,
    "neutral_watch": 0.015,
    "insufficient_sample": 0.000,
    "shrink_or_suppress": -0.035,
    "scout_required": -0.015,
    "not_reviewed": 0.000,
}

PICK_BUCKET_BONUS = {
    "round_1": 0.000,
    "round_2": 0.018,
    "top_100_day_2": 0.030,
    "early_day_3": 0.035,
    "late_day_3": 0.020,
    "udfa": -0.010,
    "unknown": 0.000,
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() < 4 or v.nunique(dropna=True) < 2:
        return pd.Series(0.50, index=values.index)
    pct = v.rank(pct=True)
    if not higher_is_better:
        pct = 1.0 - pct + (1.0 / v.notna().sum())
    return pct.fillna(0.50).clip(0.01, 0.99)


def pick_bucket(pick: pd.Series) -> pd.Series:
    p = pd.to_numeric(pick, errors="coerce")
    out = pd.Series("unknown", index=pick.index, dtype="object")
    out[p.le(32)] = "round_1"
    out[p.gt(32) & p.le(64)] = "round_2"
    out[p.gt(64) & p.le(100)] = "top_100_day_2"
    out[p.gt(100) & p.le(150)] = "early_day_3"
    out[p.gt(150) & p.le(262)] = "late_day_3"
    out[p.gt(262)] = "udfa"
    return out


def merge_cfbd(board: pd.DataFrame, cfbd_path: Path) -> pd.DataFrame:
    cfbd = read_csv(cfbd_path)
    if cfbd.empty:
        return board
    keys = ["Year", "Player"]
    if not set(keys).issubset(cfbd.columns):
        return board
    feature_cols = [c for c in cfbd.columns if c.startswith("cfbd_")]
    if not feature_cols:
        return board
    keep = keys + feature_cols
    return board.merge(cfbd[keep].drop_duplicates(keys), on=keys, how="left")


def group_score(out: pd.DataFrame, cols: Iterable[tuple[str, bool]]) -> pd.Series:
    pieces = []
    for col, hib in cols:
        if col in out.columns:
            pieces.append(percentile(out[col], hib))
    if not pieces:
        return pd.Series(0.50, index=out.index)
    return pd.concat(pieces, axis=1).mean(axis=1).fillna(0.50).clip(0.01, 0.99)


def add_production_scores(out: pd.DataFrame) -> pd.DataFrame:
    pos = out.get("pos_g", pd.Series("OTH", index=out.index)).astype(str)
    out["prospect_production_score"] = 0.50
    out["prospect_production_signal"] = "profile_only"

    qb = pos.eq("QB")
    out["qb_pass_efficiency_score"] = np.nan
    out["qb_creation_score"] = np.nan
    if qb.any():
        # 2024 lesson: an efficiency-only screen greenlit clean low-volume
        # profiles (McCarthy/Nix) while missing dual-threat creation value
        # (Daniels). QB production is now two explicit sub-scores.
        pass_score = group_score(out.loc[qb], [
            ("cfbd_pass_int_rate", False),
            ("cfbd_pass_td_rate", True),
            ("cfbd_pass_ypa", True),
            ("cfbd_final_pass_ypa", True),
        ])
        create_score = group_score(out.loc[qb], [
            ("cfbd_rush_ypc", True),
            ("cfbd_final_rush_ypc", True),
            ("cfbd_best_total_yards", True),
            ("cfbd_total_td", True),
        ])
        out.loc[qb, "qb_pass_efficiency_score"] = pass_score.round(4)
        out.loc[qb, "qb_creation_score"] = create_score.round(4)
        out.loc[qb, "prospect_production_score"] = (0.65 * pass_score + 0.35 * create_score)
        out.loc[qb, "prospect_production_signal"] = "qb_efficiency_plus_creation"

    skill = pos.isin(["RB", "WR", "TE"])
    if skill.any():
        skill_score = group_score(out.loc[skill], [
            ("cfbd_final_scrimmage_yards_per_touch", True),
            ("cfbd_scrimmage_yards_per_touch", True),
            ("cfbd_rec_ypr", True),
            ("cfbd_final_rec_ypr", True),
            ("cfbd_td_per_touch", True),
        ])
        out.loc[skill, "prospect_production_score"] = skill_score
        out.loc[skill, "prospect_production_signal"] = "skill_efficiency"

    defense = pos.isin(["EDGE", "DT", "LB", "DB"])
    if defense.any():
        def_score = group_score(out.loc[defense], [
            ("cfbd_def_playmaking", True),
            ("cfbd_final_def_playmaking", True),
            ("cfbd_sack_tfl_per_tackle", True),
        ])
        out.loc[defense, "prospect_production_score"] = def_score
        out.loc[defense, "prospect_production_signal"] = "defensive_playmaking"

    return out


def add_cautions(out: pd.DataFrame) -> pd.DataFrame:
    seasons = num(out, "cfbd_seasons")
    volume = percentile(num(out, "cfbd_best_total_yards"), True)
    tds = percentile(num(out, "cfbd_total_td"), True)
    prod = num(out, "prospect_production_score", 0.50)
    qb_pass = num(out, "qb_pass_efficiency_score")
    qb_create = num(out, "qb_creation_score")
    cautions: list[str] = []
    for i in out.index:
        flags = []
        if (
            pd.notna(qb_pass.loc[i]) and pd.notna(qb_create.loc[i])
            and qb_pass.loc[i] >= 0.70 and qb_create.loc[i] <= 0.45
        ):
            # Clean but one-dimensional: efficiency without volume/creation.
            flags.append("one_dimensional_efficiency_profile")
        if pd.notna(seasons.loc[i]) and seasons.loc[i] >= 5:
            flags.append("long_college_exposure")
        if prod.loc[i] < 0.40 and (volume.loc[i] >= 0.75 or tds.loc[i] >= 0.75):
            flags.append("volume_over_efficiency")
        if prod.loc[i] < 0.35:
            flags.append("production_caution")
        cautions.append(";".join(flags) if flags else "none")
    out["prospect_caution_flags"] = cautions
    out["prospect_caution_count"] = [0 if x == "none" else len(x.split(";")) for x in cautions]
    return out


def lens_call(row: pd.Series) -> str:
    score = float(row.get("prospect_lens_score", 0.50))
    edge = float(row.get("front_office_edge", 0.0)) if pd.notna(row.get("front_office_edge", np.nan)) else 0.0
    prod = float(row.get("prospect_production_score", 0.50))
    caution = int(row.get("prospect_caution_count", 0) or 0)
    pos = str(row.get("pos_g", ""))
    recent = bool(row.get("prospect_is_recent", False))
    if not recent:
        return "historical_result"
    if caution >= 2 and score < 0.58:
        return "avoid_risk"
    if pos == "QB":
        qb_pass = row.get("qb_pass_efficiency_score", np.nan)
        qb_create = row.get("qb_creation_score", np.nan)
        balanced = (
            pd.notna(qb_pass) and pd.notna(qb_create)
            and float(qb_pass) >= 0.45 and float(qb_create) >= 0.45
        )
        efficiency_only = (
            pd.notna(qb_pass) and pd.notna(qb_create)
            and float(qb_pass) >= 0.70 and float(qb_create) <= 0.45
        )
        # Greenlight requires strong score, strong production, balanced
        # passing AND creation, zero caution flags, and no efficiency-only
        # shape. Clean-but-limited profiles cannot clear the bar.
        if score >= 0.72 and prod >= 0.55 and balanced and caution == 0 and not efficiency_only:
            return "qb_model_greenlight"
        if score >= 0.60:
            return "qb_model_review"
    if score >= 0.78 and edge >= 0.020 and prod >= 0.52:
        return "priority_target"
    if score >= 0.68 and prod >= 0.48:
        return "starter_path"
    if score >= 0.62 and edge >= 0.015:
        return "value_development"
    if score >= 0.58:
        return "hold_grade"
    if edge <= -0.035 or caution >= 1:
        return "fade_risk"
    return "late_watch"


def lens_confidence(row: pd.Series) -> str:
    recent = bool(row.get("prospect_is_recent", False))
    if not recent:
        return "historical"
    coverage = int(row.get("prospect_signal_count", 0) or 0)
    prod = float(row.get("prospect_production_score", 0.50))
    trust = str(row.get("position_trust_label", ""))
    if coverage >= 4 and trust in {"candidate_for_more_trust", "track_but_do_not_promote", "neutral_watch"} and prod != 0.50:
        return "medium_plus"
    if coverage >= 3:
        return "medium"
    if coverage >= 2:
        return "low_plus"
    return "low"


def build_lens(board: pd.DataFrame, cfbd_path: Path = CFBD_PATH) -> pd.DataFrame:
    if board.empty:
        raise ValueError("Cannot build Prospect Lens for an empty board.")
    out = merge_cfbd(board.copy(), cfbd_path)
    year = num(out, "Year")
    out["prospect_is_recent"] = year.ge(RECENT_START_YEAR) | num(out, "y").isna()

    base = num(out, "front_office_score")
    base = base.fillna(num(out, "apex_conservative_050")).fillna(num(out, "apex")).fillna(num(out, "exp_at_pick")).fillna(0.50)
    edge = num(out, "front_office_edge").fillna(num(out, "conservative_surplus_050")).fillna(num(out, "surplus")).fillna(0.0)
    trust = out.get("position_trust_label", pd.Series("not_reviewed", index=out.index)).astype(str)
    bucket = out.get("pick_bucket", pick_bucket(out.get("Pick", pd.Series(np.nan, index=out.index)))).astype(str)

    out = add_production_scores(out)
    out = add_cautions(out)

    prod = num(out, "prospect_production_score", 0.50).fillna(0.50)
    trust_adj = trust.map(TRUST_BONUS).fillna(0.0)
    bucket_adj = bucket.map(PICK_BUCKET_BONUS).fillna(0.0)
    edge_adj = edge.clip(-0.12, 0.12) * 0.60
    caution_penalty = num(out, "prospect_caution_count", 0).fillna(0).clip(0, 3) * 0.025

    # The score is intentionally conservative: 60% validated profile/market, 25%
    # production/role confirmation, 15% edge/trust/bucket context.
    lens = (0.60 * base) + (0.25 * prod) + edge_adj + trust_adj + bucket_adj - caution_penalty
    out["prospect_lens_score"] = lens.clip(0.01, 0.99).round(4)

    signal_count = (
        base.notna().astype(int)
        + edge.notna().astype(int)
        + prod.notna().astype(int)
        + trust.notna().astype(int)
    )
    cfbd_signal = pd.Series(0, index=out.index)
    for c in ["cfbd_pass_int_rate", "cfbd_final_scrimmage_yards_per_touch", "cfbd_def_playmaking"]:
        if c in out.columns:
            cfbd_signal = cfbd_signal + pd.to_numeric(out[c], errors="coerce").notna().astype(int)
    out["prospect_signal_count"] = (signal_count + cfbd_signal.clip(0, 1)).astype(int)
    out["prospect_lens_confidence"] = out.apply(lens_confidence, axis=1)
    out["prospect_lens_call"] = out.apply(lens_call, axis=1)
    out["prospect_lens_status"] = np.where(
        out["prospect_is_recent"],
        "recent_class_model_layer_not_final_scouting_grade",
        "historical_reference",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default=str(BOARD_PATH))
    parser.add_argument("--out", default=None, help="Defaults to overwriting --board")
    parser.add_argument("--cfbd", default=str(CFBD_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--recent", default=str(RECENT_NOTES_PATH))
    args = parser.parse_args()

    board_path = Path(args.board)
    out_path = Path(args.out) if args.out else board_path
    board = read_csv(board_path)
    scored = build_lens(board, Path(args.cfbd))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.round(4).to_csv(out_path, index=False)

    recent = scored[scored["prospect_is_recent"]].copy()
    recent_cols = [c for c in [
        "Year", "Player", "Pos", "pos_g", "College", "Pick",
        "prospect_lens_score", "prospect_lens_call", "prospect_lens_confidence",
        "prospect_production_score", "prospect_production_signal", "prospect_caution_flags",
        "front_office_edge", "front_office_call", "position_trust_label",
    ] if c in recent.columns]
    Path(args.recent).parent.mkdir(parents=True, exist_ok=True)
    recent[recent_cols].sort_values(["Year", "prospect_lens_score"], ascending=[False, False]).to_csv(args.recent, index=False)

    report = {
        "rows": int(len(scored)),
        "recent_rows": int(len(recent)),
        "calls": scored.loc[scored["prospect_is_recent"], "prospect_lens_call"].value_counts(dropna=False).to_dict(),
        "confidence": scored.loc[scored["prospect_is_recent"], "prospect_lens_confidence"].value_counts(dropna=False).to_dict(),
        "cfbd_available_rows": int(sum(scored.get("cfbd_seasons", pd.Series(index=scored.index, dtype=float)).notna())) if "cfbd_seasons" in scored.columns else 0,
        "score_contract": "Prospect Lens is a recent-class model layer. It is not a final scouting grade and does not use media/mock buzz.",
        "uses_paid_data_or_apis": False,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
