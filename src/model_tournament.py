"""Model tournament: compare candidate pre-draft scores against baselines.

Walk-forward on mature classes only (default 2011-2021, outcomes through 2024).
No 2024+ partial NFL results are used for evaluation. Candidates are existing
pre-draft score columns / blends; challengers promote only via gates in
promotion_gate().
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MATURE = range(2011, 2022)


def load():
    df = pd.read_csv(ROOT / "data" / "apex_board.csv", low_memory=False)
    con = pd.read_csv(ROOT / "data" / "consensus" / "consensus_board.csv")
    df = df.merge(
        con[["Year", "Player", "consensus_rank"]].drop_duplicates(["Year", "Player"]),
        on=["Year", "Player"],
        how="left",
    )
    for c in [
        "y", "Pick", "apex", "apex_conservative_025", "apex_conservative_050",
        "apex_conservative_075", "front_office_score", "prospect_production_score",
        "apex_pff", "consensus_rank", "exp_at_pick",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def candidates(df):
    pp = df["apex_conservative_050"].rank(pct=True) * 0.75 + df["prospect_production_score"].fillna(0.5) * 0.25
    return {
        "slot_baseline": -df["Pick"],
        "consensus_baseline": -df["consensus_rank"],
        "apex_raw": df["apex"],
        "apex_conservative_050_CURRENT": df["apex_conservative_050"],
        "apex_conservative_025": df["apex_conservative_025"],
        "apex_conservative_075": df["apex_conservative_075"],
        "front_office_score": df["front_office_score"],
        "profile_plus_production": pp,
        "apex_pff": pd.to_numeric(df.get("apex_pff"), errors="coerce") if "apex_pff" in df.columns else pd.Series(np.nan, index=df.index),
    }


def prec_at(sub, score, k):
    v = sub.dropna(subset=[score, "y"])
    if len(v) < k:
        return np.nan
    top_m = set(v.nlargest(k, score).index)
    top_a = set(v.nlargest(k, "y").index)
    return len(top_m & top_a) / k


def run():
    df = load()
    rows = []
    for year in MATURE:
        sub = df[(df["Year"] == year) & df["y"].notna()].copy()
        cands = candidates(sub)
        slot_rho = spearmanr(cands["slot_baseline"], sub["y"], nan_policy="omit").statistic
        for name, score in cands.items():
            sub["_s"] = score
            v = sub.dropna(subset=["_s", "y"])
            cov = len(v) / len(sub) if len(sub) else np.nan
            rho = spearmanr(v["_s"], v["y"]).statistic if len(v) > 20 else np.nan
            rows.append({
                "year": year,
                "model": name,
                "n": len(v),
                "coverage": round(cov, 3) if pd.notna(cov) else np.nan,
                "spearman": round(rho, 4) if pd.notna(rho) else np.nan,
                "lift_vs_slot": round(rho - slot_rho, 4) if pd.notna(rho) else np.nan,
                "precision_at_32": prec_at(sub, "_s", 32),
                "precision_at_64": prec_at(sub, "_s", 64),
            })
    by_year = pd.DataFrame(rows)
    by_year.to_csv(ROOT / "reports" / "model_tournament_by_year.csv", index=False)

    lb = by_year.groupby("model").agg(
        mean_lift=("lift_vs_slot", "mean"),
        median_lift=("lift_vs_slot", "median"),
        win_rate=("lift_vs_slot", lambda s: (s > 0).mean()),
        worst_year=("lift_vs_slot", "min"),
        mean_spearman=("spearman", "mean"),
        mean_p32=("precision_at_32", "mean"),
        mean_p64=("precision_at_64", "mean"),
        mean_coverage=("coverage", "mean"),
    ).round(4).sort_values("mean_lift", ascending=False)
    lb.to_csv(ROOT / "reports" / "model_leaderboard.csv")

    cur = lb.loc["apex_conservative_050_CURRENT"]
    verdicts = {}
    for m in lb.index:
        if m in ("slot_baseline", "apex_conservative_050_CURRENT"):
            continue
        c = lb.loc[m]
        gates = {
            "mean_lift": bool(c.mean_lift > cur.mean_lift),
            "median_lift": bool(c.median_lift > cur.median_lift),
            "win_rate": bool(c.win_rate >= cur.win_rate),
            "worst_year": bool(c.worst_year >= cur.worst_year),
            "practical_p32_or_p64": bool(c.mean_p32 > cur.mean_p32 or c.mean_p64 > cur.mean_p64),
            "coverage_ok": bool(c.mean_coverage >= 0.90),
        }
        verdicts[m] = {"gates": gates, "promote": all(gates.values())}

    # LEAKAGE GUARD: stored mature-year score columns are IN-SAMPLE fits from
    # models trained on those years. They audit ranking shape only and can
    # NEVER support promotion. Authoritative walk-forward evidence is
    # reports/rolling_backtest_summary.csv (train <= year-1, test year).
    roll = pd.read_csv(ROOT / "reports" / "rolling_backtest_summary.csv")
    wf = {
        "apex_raw_vs_pick": {
            "mean": round(float(roll["delta_raw_vs_pick_spearman_drafted"].mean()), 4),
            "median": round(float(roll["delta_raw_vs_pick_spearman_drafted"].median()), 4),
            "win_rate": round(float((roll["delta_raw_vs_pick_spearman_drafted"] > 0).mean()), 3),
            "worst": round(float(roll["delta_raw_vs_pick_spearman_drafted"].min()), 4),
        },
        "market_vs_pick": {"mean": round(float(roll["delta_market_vs_pick_spearman_drafted"].mean()), 4)},
        "apex_plus_vs_pick": {
            "mean": round(float(roll["delta_plus_vs_pick_spearman_drafted"].mean()), 4),
            "worst": round(float(roll["delta_plus_vs_pick_spearman_drafted"].min()), 4),
        },
    }
    report = {
        "mature_years": [int(y) for y in MATURE],
        "baseline": "apex_conservative_050_CURRENT",
        "baseline_row": cur.to_dict(),
        "in_sample_audit_verdicts_VOID_FOR_PROMOTION": verdicts,
        "walk_forward_authoritative": wf,
        "promoted": [],
        "promotion_reason": "Stored mature-year scores are in-sample; every challenger promotion is voided. True walk-forward (rolling backtest) shows apex_raw ~+0.014 mean lift, 9/11 win years; apex_plus is negative out-of-time and stays demoted.",
        "notes": [
            "profile_plus_production is neutral-imputed for ~90% of mature rows (CFBD coverage ~10%); any apparent lift is not a validated production edge.",
            "No 2024+ outcomes used.",
        ],
    }
    (ROOT / "reports" / "model_tournament_report.json").write_text(json.dumps(report, indent=2))
    print(lb.to_string())
    print("PROMOTED:", report["promoted"] or "none")


if __name__ == "__main__":
    run()
