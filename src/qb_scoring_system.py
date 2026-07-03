"""QB-only no-noise scoring system for APEX research.

This script scores QB prospects from measurable college/profile fields only.
It intentionally excludes media consensus, mock drafts, and narrative scouting
labels. Missing fields are neutralized and reported so the score is auditable.

Typical use:
    python src/qb_scoring_system.py \
      --input data/production/qb_production.csv \
      --out reports/qb_trait_scores.csv \
      --report reports/qb_scoring_system_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline import ROOT

DEFAULT_INPUT = ROOT / "data" / "production" / "qb_production.csv"
DEFAULT_OUT = ROOT / "reports" / "qb_trait_scores.csv"
DEFAULT_REPORT = ROOT / "reports" / "qb_scoring_system_report.json"


@dataclass(frozen=True)
class MetricSpec:
    column: str
    component: str
    direction: int = 1
    weight: float = 1.0
    description: str = ""


COMPONENT_WEIGHTS = {
    "efficiency_context": 22,
    "accuracy": 16,
    "pressure_sack_avoidance": 16,
    "decision_risk": 14,
    "creation_explosiveness": 12,
    "experience_sample": 8,
    "mobility": 7,
    "measurable_thresholds": 5,
}

# Columns listed here are intentionally measurable. They can be supplied by
# qb_production.csv, PFF-derived CSVs, CFBD PPA/usage outputs, or manual verified
# combine/measurement files. If absent, they are neutral and listed in coverage.
METRICS = [
    MetricSpec("qb_final_qbr", "efficiency_context", 1, 1.0, "Final-season ESPN QBR or equivalent opponent/context-adjusted QB efficiency."),
    MetricSpec("qb_best_qbr", "efficiency_context", 1, 0.7, "Peak QBR; useful but less stable than final/career forms."),
    MetricSpec("qb_epa_per_play", "efficiency_context", 1, 1.0, "Career EPA per QB play/dropback/action play."),
    MetricSpec("qb_final_epa_per_play", "efficiency_context", 1, 0.9, "Final-season EPA efficiency."),
    MetricSpec("qb_pff_pass_grade", "efficiency_context", 1, 0.8, "Career PFF passing grade, if locally available."),
    MetricSpec("qb_pff_pass_grade_final", "efficiency_context", 1, 0.8, "Final-season PFF passing grade, if locally available."),

    MetricSpec("qb_cpoe", "accuracy", 1, 1.0, "Completion percentage over expected."),
    MetricSpec("qb_adj_completion_pct", "accuracy", 1, 0.9, "Adjusted completion percentage / accuracy percent."),
    MetricSpec("qb_accuracy_percent", "accuracy", 1, 0.9, "Charted accuracy percent."),
    MetricSpec("qb_final_accuracy_percent", "accuracy", 1, 0.8, "Final-season charted accuracy percent."),

    MetricSpec("qb_pressure_to_sack_rate", "pressure_sack_avoidance", -1, 1.0, "Share of pressures converted into sacks; lower is better."),
    MetricSpec("qb_sack_rate", "pressure_sack_avoidance", -1, 0.9, "Sacks per dropback/action play; lower is better."),
    MetricSpec("qb_sack_epa_per_play", "pressure_sack_avoidance", 1, 0.8, "EPA impact from sacks; less negative is better."),
    MetricSpec("qb_avg_time_to_throw", "pressure_sack_avoidance", -1, 0.35, "Time to throw; used lightly because scheme can dominate."),

    MetricSpec("qb_turnover_worthy_play_rate", "decision_risk", -1, 1.0, "Turnover-worthy play rate; lower is better."),
    MetricSpec("qb_interception_rate", "decision_risk", -1, 0.6, "Interception rate; lower is better, but noisier than TWP."),
    MetricSpec("qb_final_twp_rate", "decision_risk", -1, 0.8, "Final-season turnover-worthy play rate."),

    MetricSpec("qb_big_time_throw_rate", "creation_explosiveness", 1, 0.9, "Big-time throw rate / high-difficulty creation."),
    MetricSpec("qb_first_down_rate", "creation_explosiveness", 1, 0.8, "First downs per attempt/dropback."),
    MetricSpec("qb_positive_epa_percent", "creation_explosiveness", 1, 0.8, "Share of positive EPA plays."),
    MetricSpec("qb_deep_ypa", "creation_explosiveness", 1, 0.5, "Deep-pass efficiency, when available."),

    MetricSpec("qb_career_plays", "experience_sample", 1, 1.0, "Career QB action plays/dropbacks; protects against tiny samples."),
    MetricSpec("qb_starts", "experience_sample", 1, 1.0, "College starts."),
    MetricSpec("qb_seasons", "experience_sample", 1, 0.5, "Seasons with meaningful QB data."),
    MetricSpec("age", "experience_sample", -1, 0.5, "Draft age; younger production receives small preference."),

    MetricSpec("qb_run_epa_per_play", "mobility", 1, 1.0, "Rushing EPA per play."),
    MetricSpec("qb_rush_epa_per_game", "mobility", 1, 0.9, "Rushing EPA per game."),
    MetricSpec("qb_scramble_rate", "mobility", 1, 0.5, "Scramble rate; useful as creation/floor context, not a main passing substitute."),

    MetricSpec("height", "measurable_thresholds", 1, 0.4, "Height threshold; a small gate, not a major driver."),
    MetricSpec("weight", "measurable_thresholds", 1, 0.35, "Frame threshold; a small gate, not a major driver."),
    MetricSpec("hand_size", "measurable_thresholds", 1, 0.25, "Hand-size threshold if available; small penalty/gate only."),
]


RISK_FLAGS = {
    "small_sample": ("qb_career_plays", "lt", 800),
    "very_small_sample": ("qb_career_plays", "lt", 450),
    "older_prospect": ("age", "gt", 24.0),
    "high_pressure_to_sack": ("qb_pressure_to_sack_rate", "gt", 20.0),
    "high_sack_rate": ("qb_sack_rate", "gt", 7.5),
    "high_turnover_risk": ("qb_turnover_worthy_play_rate", "gt", 3.8),
    "low_creation": ("qb_big_time_throw_rate", "lt", 4.0),
    "undersized_height": ("height", "lt", 72.0),
    "light_frame": ("weight", "lt", 205.0),
}


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def percentile_score(series: pd.Series, direction: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 3 or values.nunique(dropna=True) < 2:
        return pd.Series(50.0, index=series.index)
    rank = values.rank(pct=True)
    if direction < 0:
        rank = 1.0 - rank + (1.0 / values.notna().sum())
    return (rank * 100.0).fillna(50.0).clip(1, 99)


def component_score(df: pd.DataFrame, specs: Iterable[MetricSpec]) -> tuple[pd.Series, dict]:
    scores = []
    weights = []
    used = []
    missing = []
    for spec in specs:
        if spec.column not in df.columns:
            missing.append(spec.column)
            continue
        values = pd.to_numeric(df[spec.column], errors="coerce")
        if values.notna().sum() < 3 or values.nunique(dropna=True) < 2:
            missing.append(spec.column)
            continue
        scores.append(percentile_score(values, spec.direction) * spec.weight)
        weights.append(spec.weight)
        used.append(spec.column)
    if not scores:
        return pd.Series(50.0, index=df.index), {"used": [], "missing_or_unusable": missing, "coverage": 0.0}
    total = sum(weights)
    out = sum(scores) / total
    coverage = len(used) / max(1, len(used) + len(missing))
    return out.clip(1, 99), {"used": used, "missing_or_unusable": missing, "coverage": coverage}


def add_risk_flags(out: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for _, row in out.iterrows():
        player_flags = []
        for name, (col, op, threshold) in RISK_FLAGS.items():
            if col not in out.columns or pd.isna(row.get(col)):
                continue
            val = float(row[col])
            if (op == "lt" and val < threshold) or (op == "gt" and val > threshold):
                player_flags.append(name)
        flags.append(";".join(player_flags))
    out["qb_risk_flags"] = flags
    out["qb_risk_flag_count"] = [0 if not f else len(f.split(";")) for f in flags]
    return out


def score_qbs(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {"status": "skipped_empty_input"}
    out = df.copy()
    report = {"status": "ok", "components": {}, "component_weights": COMPONENT_WEIGHTS}
    total = pd.Series(0.0, index=out.index)
    total_weight = 0.0
    for component, weight in COMPONENT_WEIGHTS.items():
        specs = [s for s in METRICS if s.component == component]
        score, meta = component_score(out, specs)
        out[f"qb_{component}_score"] = score.round(2)
        total += score * weight
        total_weight += weight
        report["components"][component] = meta
    out["qb_trait_score"] = (total / total_weight).round(2).clip(1, 99)
    out = add_risk_flags(out)

    def tier(value: float) -> str:
        if value >= 85:
            return "blue_chip_trait_profile"
        if value >= 75:
            return "starter_trait_profile"
        if value >= 65:
            return "developmental_starter_traits"
        if value >= 55:
            return "backup_or_system_profile"
        return "low_trait_score"

    out["qb_trait_tier"] = out["qb_trait_score"].map(tier)
    report["rows"] = int(len(out))
    report["promotion_contract"] = "No QB score is promoted until rolling out-of-time QB-only validation passes. This score excludes media/mock noise."
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    df = read_input(Path(args.input))
    scored, report = score_qbs(df)
    out_path = Path(args.out)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report.get("status"), "rows": int(len(scored)), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
