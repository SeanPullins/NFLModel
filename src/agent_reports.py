"""Deterministic feature, calibration, position-edge, and miss reports.

Mature years: 2011-2021. Bucket assumptions: star y>=0.90,
starter y>=0.60, bust y<=0.30.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
MATURE_START, MATURE_END = 2011, 2021

board = pd.read_csv(ROOT / "data" / "apex_board.csv", low_memory=False)
mature_mask = board["Year"].between(MATURE_START, MATURE_END)
recent_mask = board["Year"] >= 2024
mature = board[mature_mask & pd.to_numeric(board["y"], errors="coerce").notna()].copy()
for col in ["y", "Pick", "apex_conservative_050", "p_star", "p_starter", "p_bust"]:
    mature[col] = pd.to_numeric(mature[col], errors="coerce")

# Feature coverage.
display_only = {"y", "CarAV", "apex_live", "qb_lens_warning"}
coverage_rows = []
for col in board.columns:
    raw = board[col]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().any():
        mature_cov = numeric[mature_mask].notna().mean()
        recent_cov = numeric[recent_mask].notna().mean()
    else:
        mature_cov = raw[mature_mask].astype(str).ne("nan").mean()
        recent_cov = raw[recent_mask].astype(str).ne("nan").mean()
    recent_only = bool(recent_cov > 0.30 and mature_cov < 0.15)
    coverage_rows.append({
        "feature": col,
        "mature_coverage": round(float(mature_cov), 3),
        "recent_coverage": round(float(recent_cov), 3),
        "display_only_outcome_related": col in display_only,
        "recent_only": recent_only,
        "allowed_in": "display_only" if col in display_only else ("research_only" if recent_only else "public_or_challenger"),
    })
coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(REPORTS / "feature_coverage.csv", index=False)
coverage[coverage.display_only_outcome_related].to_csv(REPORTS / "feature_leakage_risk.csv", index=False)
(REPORTS / "feature_quality_report.json").write_text(json.dumps({
    "mature_years": [MATURE_START, MATURE_END],
    "recent_only_features_cannot_validate": coverage[coverage.recent_only].feature.tolist(),
    "display_only_outcome_related": sorted(display_only),
    "rule": "Recent-only features stay research/QB-Lens only until mature-class validation exists.",
}, indent=2))

# Calibration.
mature["star"] = (mature["y"] >= 0.90).astype(int)
mature["starter"] = (mature["y"] >= 0.60).astype(int)
mature["bust"] = (mature["y"] <= 0.30).astype(int)

def calibration_stats(pred_col, actual_col, frame):
    data = frame.dropna(subset=[pred_col, actual_col])
    if len(data) < 50:
        return None
    brier = float(((data[pred_col] - data[actual_col]) ** 2).mean())
    buckets = pd.cut(data[pred_col], np.linspace(0, 1, 11), include_lowest=True)
    rel = data.groupby(buckets, observed=True).agg(pred=(pred_col, "mean"), actual=(actual_col, "mean"), n=(actual_col, "size"))
    ece = float((rel["n"] / len(data) * (rel.pred - rel.actual).abs()).sum())
    return {
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "pred_rate": round(float(data[pred_col].mean()), 4),
        "actual_rate": round(float(data[actual_col].mean()), 4),
    }

overall = {name: calibration_stats(f"p_{name}", name, mature) for name in ["star", "starter", "bust"]}
reliability = []
for name in ["star", "starter", "bust"]:
    data = mature.dropna(subset=[f"p_{name}", name])
    buckets = pd.cut(data[f"p_{name}"], np.linspace(0, 1, 11), include_lowest=True)
    out = data.groupby(buckets, observed=True).agg(pred=(f"p_{name}", "mean"), actual=(name, "mean"), n=(name, "size")).reset_index(drop=True)
    out["bucket_target"] = name
    reliability.append(out)
pd.concat(reliability).round(4).to_csv(REPORTS / "reliability_buckets.csv", index=False)

pos_cal = []
for pos, group in mature.groupby("pos_g"):
    pos_cal.append({
        "pos_g": pos,
        "n": len(group),
        "star_pred": group["p_star"].mean(),
        "star_actual": group["star"].mean(),
        "bust_pred": group["p_bust"].mean(),
        "bust_actual": group["bust"].mean(),
    })
pd.DataFrame(pos_cal).round(4).to_csv(REPORTS / "calibration_by_position.csv", index=False)

round_cal = []
for rnd, group in mature.groupby("Rnd"):
    round_cal.append({
        "Rnd": rnd,
        "n": len(group),
        "star_pred": group["p_star"].mean(),
        "star_actual": group["star"].mean(),
        "bust_pred": group["p_bust"].mean(),
        "bust_actual": group["bust"].mean(),
    })
pd.DataFrame(round_cal).round(4).to_csv(REPORTS / "calibration_by_round.csv", index=False)
(REPORTS / "calibration_report.json").write_text(json.dumps({
    "bucket_defs": {"star": "y>=0.90", "starter": "y>=0.60", "bust": "y<=0.30"},
    "mature_years": [MATURE_START, MATURE_END],
    "overall": overall,
    "promotion": "No recalibration promoted this run; measure-first. If ECE is high, fit isotonic on prior-year folds in a follow-up.",
}, indent=2))

# Position edge.
pos_rows = []
for pos, group in mature.groupby("pos_g"):
    lifts = []
    for _, year_group in group.groupby("Year"):
        year_group = year_group.dropna(subset=["Pick", "apex_conservative_050", "y"])
        if len(year_group) < 12:
            continue
        slot = spearmanr(-year_group["Pick"], year_group["y"]).statistic
        model = spearmanr(year_group["apex_conservative_050"], year_group["y"]).statistic
        lifts.append(model - slot)
    if not lifts:
        continue
    lifts = pd.Series(lifts)
    pos_rows.append({
        "pos": pos,
        "years": len(lifts),
        "mean_lift": round(lifts.mean(), 4),
        "win_rate": round((lifts > 0).mean(), 3),
        "worst_year": round(lifts.min(), 4),
        "recommendation": "position_weighting_candidate" if lifts.mean() > 0.01 and (lifts > 0).mean() >= 0.6 else ("global_model" if lifts.mean() > -0.01 else "shrink_to_market"),
    })
position_edge = pd.DataFrame(pos_rows).sort_values("mean_lift", ascending=False)
position_edge.to_csv(REPORTS / "position_edge_report.csv", index=False)
(REPORTS / "position_model_recommendations.json").write_text(position_edge.to_json(orient="records", indent=2))

# Miss audit.
mature["err"] = mature["apex_conservative_050"] - mature["y"]

def category(row):
    if row["Pick"] <= 15 and row["err"] > 0.4:
        return "market_miss_shared"
    if row["pos_g"] == "RB" and row["err"] > 0.3:
        return "position_value_distortion"
    if row["err"] > 0.35:
        return "model_miss"
    if row["err"] < -0.35:
        return "late_breakout_or_context_miss"
    return "moderate"

miss_source = mature.nlargest(25, "err")
hit_source = mature.nsmallest(25, "err")
misses = miss_source[["Year", "Player", "pos_g", "Pick", "apex_conservative_050", "y", "err"]].copy()
hits = hit_source[["Year", "Player", "pos_g", "Pick", "apex_conservative_050", "y", "err"]].copy()
misses["category"] = miss_source.apply(category, axis=1).values
misses["suggests"] = np.where(misses["category"] == "market_miss_shared", "warning_label", "review_weighting")
misses.round(3).to_csv(REPORTS / "biggest_misses.csv", index=False)
hits.round(3).to_csv(REPORTS / "biggest_hits.csv", index=False)
(REPORTS / "miss_categories.json").write_text(json.dumps(misses["category"].value_counts().to_dict(), indent=2))

print("agents 3-6 reports written")
print(json.dumps(overall, indent=2))
print(position_edge.to_string(index=False))
