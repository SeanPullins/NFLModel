#!/usr/bin/env python3
"""
APEX+ v2 audit/projection engine.

This script does NOT replace the website model. It audits the committed
`data/apex_board.csv`, runs stronger validation around the existing APEX+ formula,
and produces a safer APEX+ v2 projection layer with shrinkage, volatility, and
risk fields.

Why this exists:
- The site currently displays a strong APEX+ signal, but raw feature-generation
  code is not present in the repo.
- Therefore, we cannot prove raw `apex` is leakage-free from this repo alone.
- This script validates what can be validated from the committed CSV and creates
  an auditable projection layer that is less overconfident than the old K=3.5
  formula.

Outputs are written to `outputs/apex_v2/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "apex_board.csv"
OUT_DIR = ROOT / "outputs" / "apex_v2"

OLD_K = 3.5
K_GRID = np.round(np.arange(0.25, 4.01, 0.25), 2)

# Coarse model groups; the CSV also has pos_g, but this keeps the script robust.
POS_GROUP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "C": "OL", "OG": "OL", "OT": "OL", "G": "OL", "T": "OL",
    "DT": "DL", "DE": "EDGE", "OLB": "LB", "ILB": "LB", "LB": "LB",
    "CB": "DB", "FS": "DB", "SS": "DB", "S": "DB",
    "K": "ST", "P": "ST", "LS": "ST",
}

# Conservative dampeners. These are intentionally transparent and can be tuned
# by rolling validation. Round 1 and high-variance positions get shrunk hardest.
ROUND_DAMPENER = {
    "Round 1": 0.40,
    "Round 2": 0.60,
    "Round 3": 0.80,
    "Rounds 4-5": 1.00,
    "Rounds 6-7": 1.00,
    "Late/UDFA": 0.85,
    "Unknown": 0.65,
}

POSITION_DAMPENER = {
    "QB": 0.55,
    "OL": 0.55,
    "DL": 0.60,
    "EDGE": 0.60,
    "DB": 0.70,
    "TE": 0.75,
    "LB": 0.80,
    "WR": 0.90,
    "RB": 0.95,
    "ST": 0.50,
    "OTHER": 0.70,
}


def clip01(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(x, 0.01, 0.99)


def round_group(pick: float | int | None) -> str:
    if pd.isna(pick):
        return "Late/UDFA"
    p = float(pick)
    if p <= 32:
        return "Round 1"
    if p <= 64:
        return "Round 2"
    if p <= 100:
        return "Round 3"
    if p <= 160:
        return "Rounds 4-5"
    if p <= 224:
        return "Rounds 6-7"
    return "Late/UDFA"


def position_group(pos: str) -> str:
    if pd.isna(pos):
        return "OTHER"
    return POS_GROUP.get(str(pos).upper(), "OTHER")


def tier(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x >= 0.85:
        return "Star"
    if x >= 0.70:
        return "Starter"
    if x >= 0.45:
        return "Contributor"
    return "Minimal"


def spearman(a: pd.Series, b: pd.Series) -> float:
    frame = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["a"].corr(frame["b"], method="spearman"))


def mae(pred: pd.Series, actual: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if len(frame) == 0:
        return np.nan
    return float((frame["pred"] - frame["actual"]).abs().mean())


def rmse(pred: pd.Series, actual: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if len(frame) == 0:
        return np.nan
    return float(np.sqrt(((frame["pred"] - frame["actual"]) ** 2).mean()))


def edge_hit_rate(pred: pd.Series, market: pd.Series, actual: pd.Series, min_edge: float = 0.02) -> float:
    frame = pd.DataFrame({"pred": pred, "market": market, "actual": actual}).dropna()
    if len(frame) == 0:
        return np.nan
    edge = frame["pred"] - frame["market"]
    actual_delta = frame["actual"] - frame["market"]
    frame = frame[edge.abs() >= min_edge].copy()
    if len(frame) == 0:
        return np.nan
    edge = frame["pred"] - frame["market"]
    actual_delta = frame["actual"] - frame["market"]
    return float((((edge > 0) & (actual_delta > 0)) | ((edge < 0) & (actual_delta < 0))).mean())


def current_apex_plus(apex: pd.Series, market: pd.Series, k: float = OLD_K) -> pd.Series:
    return pd.Series(clip01(market + k * (apex - market)), index=apex.index)


def build_market_curve(train: pd.DataFrame) -> Dict[float, float]:
    """Build prior-year market baseline by exact pick, with round fallback handled elsewhere."""
    base = train.dropna(subset=["Pick", "y"]).copy()
    return base.groupby("Pick")["y"].median().to_dict()


def market_from_history(test: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Prior-year market baseline for a test year.

    Exact pick median from prior years first. If unavailable, nearby pick window.
    If unavailable, same round group median. If unavailable, existing exp_at_pick.
    """
    train = train.dropna(subset=["Pick", "y"]).copy()
    test = test.copy()
    exact = train.groupby("Pick")["y"].median().to_dict()
    round_median = train.groupby("round_group")["y"].median().to_dict()

    def lookup(row: pd.Series) -> float:
        pick = row.get("Pick")
        if pd.isna(pick):
            if not pd.isna(row.get("exp_at_pick")):
                return row.get("exp_at_pick")
            return np.nan
        if pick in exact:
            return exact[pick]
        nearby = train[(train["Pick"] >= pick - 3) & (train["Pick"] <= pick + 3)]["y"].dropna()
        if len(nearby):
            return float(nearby.median())
        rg = row.get("round_group", round_group(pick))
        if rg in round_median:
            return float(round_median[rg])
        return row.get("exp_at_pick") if not pd.isna(row.get("exp_at_pick")) else np.nan

    return test.apply(lookup, axis=1)


def sigma_by_pos(train: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    frame = train.dropna(subset=["y", "exp_at_pick"]).copy()
    frame["resid"] = frame["y"] - frame["exp_at_pick"]
    for pg, g in frame.groupby("pos_group_clean"):
        if len(g) >= 25:
            out[pg] = float(max(g["resid"].std(), 0.08))
    out["OTHER"] = float(max(frame["resid"].std(), 0.10)) if len(frame) else 0.18
    return out


def apex_v2_projection(apex: pd.Series, market: pd.Series, pos_group_s: pd.Series,
                       round_group_s: pd.Series, sigma_map: Dict[str, float], k_base: float = 1.0) -> pd.Series:
    edge = apex - market
    vals: List[float] = []
    for idx in apex.index:
        a = apex.loc[idx]
        m = market.loc[idx]
        if pd.isna(a) or pd.isna(m):
            vals.append(np.nan)
            continue
        pg = pos_group_s.loc[idx]
        rg = round_group_s.loc[idx]
        sigma = sigma_map.get(pg, sigma_map.get("OTHER", 0.18))
        pos_d = POSITION_DAMPENER.get(pg, POSITION_DAMPENER["OTHER"])
        round_d = ROUND_DAMPENER.get(rg, ROUND_DAMPENER["Unknown"])
        e = a - m
        edge_d = 1.0 / (1.0 + (abs(e) / 0.25) ** 2)
        adjusted_k = k_base * pos_d * round_d * edge_d
        shrunken_edge = (adjusted_k * e) / (1.0 + abs(e) / sigma)
        vals.append(float(clip01(m + shrunken_edge)))
    return pd.Series(vals, index=apex.index)


def summarize(pred: pd.Series, market: pd.Series, actual: pd.Series, label: str) -> Dict[str, float | str | int]:
    frame = pd.DataFrame({"pred": pred, "market": market, "actual": actual}).dropna()
    if len(frame) == 0:
        return {"label": label, "n": 0}
    err = frame["pred"] - frame["actual"]
    return {
        "label": label,
        "n": len(frame),
        "spearman": spearman(frame["pred"], frame["actual"]),
        "market_spearman": spearman(frame["market"], frame["actual"]),
        "lift": spearman(frame["pred"], frame["actual"]) - spearman(frame["market"], frame["actual"]),
        "mae_pts": mae(frame["pred"], frame["actual"]) * 100,
        "market_mae_pts": mae(frame["market"], frame["actual"]) * 100,
        "rmse_pts": rmse(frame["pred"], frame["actual"]) * 100,
        "market_rmse_pts": rmse(frame["market"], frame["actual"]) * 100,
        "edge_hit": edge_hit_rate(frame["pred"], frame["market"], frame["actual"]),
        "miss30_rate": (err.abs() >= 0.30).mean(),
        "false_optimism30": int((err >= 0.30).sum()),
        "false_pessimism30": int((err <= -0.30).sum()),
    }


def tune_k(train: pd.DataFrame, market_col: str = "exp_at_pick") -> float:
    best_k = 1.0
    best_score = -999.0
    frame = train.dropna(subset=["apex", market_col, "y"]).copy()
    if len(frame) < 50:
        return best_k
    for k in K_GRID:
        pred = current_apex_plus(frame["apex"], frame[market_col], k)
        score = spearman(pred, frame["y"])
        if pd.notna(score) and score > best_score:
            best_score = score
            best_k = float(k)
    return best_k


def rolling_validation(df: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        train = df[(df["Year"] < year) & (df["y"].notna())].copy()
        test = df[(df["Year"] == year) & (df["y"].notna())].copy()
        if len(train) < 500 or len(test) < 20:
            continue
        # Build a truly prior market curve. This tests market creation more honestly.
        test["market_prior"] = market_from_history(test, train)
        train["market_prior"] = train["exp_at_pick"]
        old_k = OLD_K
        tuned_k = tune_k(train, "exp_at_pick")
        sigmas = sigma_by_pos(train)

        old_pred = current_apex_plus(test["apex"], test["market_prior"], old_k)
        tuned_pred = current_apex_plus(test["apex"], test["market_prior"], tuned_k)
        v2_pred = apex_v2_projection(
            test["apex"], test["market_prior"], test["pos_group_clean"], test["round_group"], sigmas, k_base=1.0
        )

        for name, pred in [
            ("old_k3_5_prior_market", old_pred),
            ("old_tuned_k_prior_market", tuned_pred),
            ("apex_v2_prior_market", v2_pred),
        ]:
            s = summarize(pred, test["market_prior"], test["y"], name)
            s.update({"year": year, "tuned_k": tuned_k})
            rows.append(s)
    return pd.DataFrame(rows)


def loyo_validation(df: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        train = df[(df["Year"] != year) & (df["y"].notna()) & (df["exp_at_pick"].notna())].copy()
        test = df[(df["Year"] == year) & (df["y"].notna()) & (df["exp_at_pick"].notna())].copy()
        if len(train) < 500 or len(test) < 20:
            continue
        tuned_k = tune_k(train, "exp_at_pick")
        sigmas = sigma_by_pos(train)
        preds = {
            "market_only": test["exp_at_pick"],
            "old_k3_5": current_apex_plus(test["apex"], test["exp_at_pick"], OLD_K),
            "old_tuned_k": current_apex_plus(test["apex"], test["exp_at_pick"], tuned_k),
            "apex_v2": apex_v2_projection(
                test["apex"], test["exp_at_pick"], test["pos_group_clean"], test["round_group"], sigmas, k_base=1.0
            ),
        }
        for name, pred in preds.items():
            s = summarize(pred, test["exp_at_pick"], test["y"], name)
            s.update({"year": year, "tuned_k": tuned_k})
            rows.append(s)
    return pd.DataFrame(rows)


def leakage_sniff(df: pd.DataFrame) -> pd.DataFrame:
    """Partial-residual style sniff test.

    It does not prove leakage. It asks: after a linear market baseline, how much
    residual signal does raw apex still carry? High residual correlation can be
    good signal, but extremely high values are a red flag requiring source-code review.
    """
    rows = []
    frame = df.dropna(subset=["apex", "exp_at_pick", "y"]).copy()
    for label, g in [("all", frame)] + [(str(int(y)), gy) for y, gy in frame.groupby("Year") if len(gy) >= 30]:
        x = g["exp_at_pick"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        a = g["apex"].to_numpy(dtype=float)
        if len(g) < 30 or np.std(x) == 0:
            continue
        # OLS y = b0 + b1*market
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X.dot(beta)
        rows.append({
            "group": label,
            "n": len(g),
            "apex_vs_y_spearman": spearman(pd.Series(a), pd.Series(y)),
            "market_vs_y_spearman": spearman(pd.Series(x), pd.Series(y)),
            "apex_vs_market_residual_spearman": spearman(pd.Series(a), pd.Series(resid)),
        })
    return pd.DataFrame(rows)


def projection_board(df: pd.DataFrame) -> pd.DataFrame:
    train = df[(df["Year"] <= 2024) & (df["y"].notna())].copy()
    future = df[df["Year"].isin([2025, 2026])].copy()
    if len(future) == 0:
        return pd.DataFrame()

    # Use official exp_at_pick if present; otherwise prior market from history.
    future["market_baseline"] = future["exp_at_pick"]
    missing = future["market_baseline"].isna()
    if missing.any():
        future.loc[missing, "market_baseline"] = market_from_history(future.loc[missing], train)

    sigmas = sigma_by_pos(train)
    future["old_apex_plus"] = current_apex_plus(future["apex"], future["market_baseline"], OLD_K)
    future["new_projection"] = apex_v2_projection(
        future["apex"], future["market_baseline"], future["pos_group_clean"], future["round_group"], sigmas, k_base=1.0
    )
    future["market_edge"] = future["apex"] - future["market_baseline"]

    # Residual distribution by pos/round from v2 on historical rows.
    train2 = train.dropna(subset=["apex", "exp_at_pick", "y"]).copy()
    train2["v2_pred"] = apex_v2_projection(
        train2["apex"], train2["exp_at_pick"], train2["pos_group_clean"], train2["round_group"], sigmas, k_base=1.0
    )
    train2["resid"] = train2["y"] - train2["v2_pred"]

    def residual_pool(row: pd.Series) -> pd.Series:
        pool = train2[(train2["pos_group_clean"] == row["pos_group_clean"]) & (train2["round_group"] == row["round_group"])]
        if len(pool) < 40:
            pool = train2[train2["pos_group_clean"] == row["pos_group_clean"]]
        if len(pool) < 40:
            pool = train2
        return pool["resid"].dropna()

    lows, highs, vols, busts, stars, starters, notes = [], [], [], [], [], [], []
    for _, row in future.iterrows():
        pool = residual_pool(row)
        pred = row["new_projection"]
        if pd.isna(pred) or len(pool) == 0:
            lows.append(np.nan); highs.append(np.nan); vols.append(np.nan); busts.append(np.nan); stars.append(np.nan); starters.append(np.nan); notes.append("Insufficient data")
            continue
        sim = clip01(pred + pool.to_numpy())
        low = float(np.quantile(sim, 0.10))
        high = float(np.quantile(sim, 0.90))
        vol = float(np.std(sim) * 100)
        bust = float((sim <= pred - 0.30).mean())
        star = float((sim >= 0.85).mean())
        starter = float((sim >= 0.70).mean())
        note_parts = []
        if bust >= 0.25:
            note_parts.append("High bust-risk band")
        if abs(row.get("market_edge", 0)) >= 0.15:
            note_parts.append("Large model/market disagreement")
        if row["pos_group_clean"] in {"QB", "OL", "DL", "EDGE", "DB"}:
            note_parts.append("High-variance position")
        if not note_parts:
            note_parts.append("Normal projection risk")
        lows.append(low); highs.append(high); vols.append(vol); busts.append(bust); stars.append(star); starters.append(starter); notes.append("; ".join(note_parts))

    future["confidence_low"] = lows
    future["confidence_high"] = highs
    future["volatility_score"] = vols
    future["bust_risk"] = busts
    future["star_probability"] = stars
    future["starter_probability"] = starters
    future["model_note"] = notes
    future["projected_tier_v2"] = future["new_projection"].map(tier)

    cols = [
        "Year", "Player", "Pos", "pos_group_clean", "College", "Pick",
        "apex", "market_baseline", "old_apex_plus", "new_projection", "market_edge",
        "confidence_low", "confidence_high", "volatility_score", "bust_risk",
        "star_probability", "starter_probability", "projected_tier_v2", "model_note",
    ]
    return future[cols].sort_values(["Year", "new_projection"], ascending=[True, False])


def write_markdown_report(df: pd.DataFrame, loyo: pd.DataFrame, rolling: pd.DataFrame,
                          sniff: pd.DataFrame, board: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# APEX+ v2 Audit Report\n")
    lines.append("Generated by `scripts/apex_v2_audit_projection.py`.\n")
    lines.append("## Transparency note\n")
    lines.append("The repo contains `data/apex_board.csv`, but this script did not find or use raw feature-generation code for `apex`, `y`, or `exp_at_pick`. Therefore, this report can validate the committed CSV behavior, but it cannot prove that raw `apex` is leakage-free.\n")

    lines.append("## Data inventory\n")
    lines.append(f"- Rows: {len(df):,}\n")
    lines.append(f"- Years: {int(df['Year'].min())}–{int(df['Year'].max())}\n")
    for col in ["apex", "exp_at_pick", "y", "CarAV", "Pick"]:
        lines.append(f"- Non-null `{col}`: {df[col].notna().sum():,}\n")

    if len(loyo):
        lines.append("\n## LOYO summary by formula\n")
        agg = loyo.groupby("label").agg(
            years=("year", "count"), spearman=("spearman", "mean"), market_spearman=("market_spearman", "mean"),
            lift=("lift", "mean"), mae_pts=("mae_pts", "mean"), edge_hit=("edge_hit", "mean"),
            miss30_rate=("miss30_rate", "mean"), false_optimism30=("false_optimism30", "sum"),
            false_pessimism30=("false_pessimism30", "sum"),
        ).reset_index()
        lines.append(agg.round(4).to_markdown(index=False))
        lines.append("\n")

    if len(rolling):
        lines.append("\n## Rolling prior-year summary by formula\n")
        agg = rolling.groupby("label").agg(
            years=("year", "count"), spearman=("spearman", "mean"), market_spearman=("market_spearman", "mean"),
            lift=("lift", "mean"), mae_pts=("mae_pts", "mean"), edge_hit=("edge_hit", "mean"),
            miss30_rate=("miss30_rate", "mean"), false_optimism30=("false_optimism30", "sum"),
            false_pessimism30=("false_pessimism30", "sum"),
        ).reset_index()
        lines.append(agg.round(4).to_markdown(index=False))
        lines.append("\n")

    if len(board):
        lines.append("\n## Future board availability\n")
        lines.append(f"- 2025/2026 rows scored: {len(board):,}\n")
        lines.append("\n### Top 25 future v2 projections\n")
        top = board.head(25).copy()
        pct_cols = ["apex", "market_baseline", "old_apex_plus", "new_projection", "market_edge", "confidence_low", "confidence_high", "bust_risk", "star_probability", "starter_probability"]
        for c in pct_cols:
            if c in top:
                top[c] = (top[c] * 100).round(1)
        lines.append(top[["Year", "Player", "Pos", "College", "Pick", "old_apex_plus", "new_projection", "market_edge", "bust_risk", "star_probability", "starter_probability", "model_note"]].to_markdown(index=False))
        lines.append("\n")
    else:
        lines.append("\n## Future board availability\n")
        lines.append("No 2025/2026 rows could be scored from the current CSV.\n")

    lines.append("\n## Recommended next step\n")
    lines.append("Publish the scripts that create `apex`, `y`, `exp_at_pick`, and `apex_board.csv`. Without those, validation can only audit the committed predictions, not the model provenance.\n")
    (OUT_DIR / "apex_v2_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    for col in ["Year", "Pick", "CarAV", "y", "apex", "exp_at_pick", "talent_resid", "surplus"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["round_group"] = df["Pick"].map(round_group)
    df["pos_group_clean"] = df["Pos"].map(position_group)
    df["old_apex_plus"] = current_apex_plus(df["apex"], df["exp_at_pick"], OLD_K)

    inventory = df.groupby("Year").agg(
        rows=("Player", "count"),
        picks=("Pick", lambda s: s.notna().sum()),
        apex=("apex", lambda s: s.notna().sum()),
        exp_at_pick=("exp_at_pick", lambda s: s.notna().sum()),
        y=("y", lambda s: s.notna().sum()),
        CarAV=("CarAV", lambda s: s.notna().sum()),
    ).reset_index()
    inventory.to_csv(OUT_DIR / "data_inventory_by_year.csv", index=False)

    tested_years = sorted(int(y) for y in df.loc[df["Year"].between(2000, 2024), "Year"].dropna().unique())
    loyo = loyo_validation(df, tested_years)
    rolling = rolling_validation(df, [y for y in tested_years if y >= 2005])
    sniff = leakage_sniff(df)
    board = projection_board(df)

    loyo.to_csv(OUT_DIR / "loyo_validation.csv", index=False)
    rolling.to_csv(OUT_DIR / "rolling_prior_validation.csv", index=False)
    sniff.to_csv(OUT_DIR / "leakage_sniff_test.csv", index=False)
    board.to_csv(OUT_DIR / "apex_v2_2025_2026_projection_board.csv", index=False)

    # Useful leaderboard cuts if future rows exist.
    if len(board):
        board.head(50).to_csv(OUT_DIR / "top50_future_v2.csv", index=False)
        board.sort_values("market_edge", ascending=False).head(50).to_csv(OUT_DIR / "best_market_edges_future_v2.csv", index=False)
        board.sort_values("market_edge", ascending=True).head(50).to_csv(OUT_DIR / "biggest_model_fades_future_v2.csv", index=False)
        board.sort_values("bust_risk", ascending=False).head(50).to_csv(OUT_DIR / "highest_bust_risk_future_v2.csv", index=False)
        board.sort_values(["starter_probability", "bust_risk"], ascending=[False, True]).head(50).to_csv(OUT_DIR / "safest_starter_future_v2.csv", index=False)

    write_markdown_report(df, loyo, rolling, sniff, board)

    print("APEX+ v2 audit complete")
    print(f"Rows: {len(df):,}")
    print(f"Years: {int(df['Year'].min())}-{int(df['Year'].max())}")
    print(f"Outputs: {OUT_DIR}")
    if len(loyo):
        print("\nLOYO summary:")
        print(loyo.groupby("label")[["spearman", "market_spearman", "lift", "mae_pts", "edge_hit", "miss30_rate"]].mean().round(4))
    if len(rolling):
        print("\nRolling prior summary:")
        print(rolling.groupby("label")[["spearman", "market_spearman", "lift", "mae_pts", "edge_hit", "miss30_rate"]].mean().round(4))
    if len(board):
        print("\nTop 10 future v2 projections:")
        cols = ["Year", "Player", "Pos", "College", "Pick", "old_apex_plus", "new_projection", "market_edge", "bust_risk", "star_probability", "starter_probability"]
        preview = board.head(10)[cols].copy()
        for c in ["old_apex_plus", "new_projection", "market_edge", "bust_risk", "star_probability", "starter_probability"]:
            preview[c] = (preview[c] * 100).round(1)
        print(preview.to_string(index=False))
    else:
        print("\nNo 2025/2026 projection rows were scoreable from the current CSV.")


if __name__ == "__main__":
    main()
