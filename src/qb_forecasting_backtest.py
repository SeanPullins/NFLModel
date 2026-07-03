"""QB-only forecasting backtest on mature classes (2011-2021).

Answers: do we have real QB edge over the market, or just better labels?
QB draft classes are ~10-14 players, so yearly Spearman is high-variance;
aggregate stats matter more than any single year.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MATURE = range(2011, 2022)

df = pd.read_csv(ROOT / "data" / "apex_board.csv", low_memory=False)
con = pd.read_csv(ROOT / "data" / "consensus" / "consensus_board.csv")
df = df.merge(
    con[["Year", "Player", "consensus_rank"]].drop_duplicates(["Year", "Player"]),
    on=["Year", "Player"],
    how="left",
)
qb = df[df["pos_g"] == "QB"].copy()
for c in [
    "y", "Pick", "apex_conservative_050", "prospect_lens_score",
    "prospect_production_score", "consensus_rank", "qb_pass_efficiency_score",
    "qb_creation_score",
]:
    qb[c] = pd.to_numeric(qb.get(c), errors="coerce")

cands = {
    "slot_baseline": lambda s: -s["Pick"],
    "consensus_baseline": lambda s: -s["consensus_rank"],
    "apex_conservative_050_CURRENT": lambda s: s["apex_conservative_050"],
    "qb_lens_challenger": lambda s: s["prospect_lens_score"],
    "qb_production_only": lambda s: s["prospect_production_score"],
}

rows = []
for year in MATURE:
    s = qb[(qb["Year"] == year) & qb["y"].notna() & qb["Pick"].notna()]
    if len(s) < 5:
        continue
    slot = spearmanr(-s["Pick"], s["y"], nan_policy="omit").statistic
    for name, fn in cands.items():
        v = pd.DataFrame({"m": fn(s), "y": s["y"]}).dropna()
        rho = spearmanr(v["m"], v["y"]).statistic if len(v) >= 5 else np.nan
        rows.append({
            "year": year,
            "model": name,
            "n_qb": len(v),
            "spearman": round(rho, 4) if pd.notna(rho) else np.nan,
            "lift_vs_slot": round(rho - slot, 4) if pd.notna(rho) else np.nan,
        })
by = pd.DataFrame(rows)
by.to_csv(ROOT / "reports" / "qb_backtest_by_year.csv", index=False)
lb = by.groupby("model").agg(
    mean_lift=("lift_vs_slot", "mean"),
    median_lift=("lift_vs_slot", "median"),
    win_rate=("lift_vs_slot", lambda x: (x > 0).mean()),
    worst_year=("lift_vs_slot", "min"),
    years=("year", "nunique"),
    mean_n=("n_qb", "mean"),
).round(4).sort_values("mean_lift", ascending=False)
lb.to_csv(ROOT / "reports" / "qb_model_leaderboard.csv")
cur = lb.loc["apex_conservative_050_CURRENT"]
promote = {}
for m in ["qb_lens_challenger", "qb_production_only"]:
    c = lb.loc[m]
    promote[m] = bool(
        c.mean_lift > cur.mean_lift
        and c.median_lift > cur.median_lift
        and c.win_rate >= cur.win_rate
        and c.worst_year >= cur.worst_year
    )
print(lb.to_string())
print("QB challenger promotion:", promote)
Path(ROOT / "reports" / "qb_promotion.json").write_text(json.dumps(
    {
        "promote": promote,
        "note": "QB production is neutral-imputed for most mature years; lens/production challengers remain explanatory QB Lens only unless gates pass.",
    },
    indent=2,
))
