"""Zero-cost forecasting sprint report for APEX.

Consumes existing validation artifacts and writes a task queue for improving
forecasting without paid APIs or unvalidated score promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

AGENTS = [
    ["Orchestrator", "GPT-5.5 Thinking + repo checks", "Decide what can be promoted vs kept experimental."],
    ["Data QA Agent", "pandas checks", "Block blank boards, leakage-prone fields, and stale generated outputs."],
    ["Validation Agent", "existing LightGBM backtests", "Measure lift, win rate, and worst-year drawdown vs draft slot."],
    ["Position Specialist Agent", "grouped diagnostics", "Find positions to trust, shrink, or mark scout-required."],
    ["Calibration Agent", "historical buckets", "Check if value/fade bands map to real outperformance."],
    ["Product/Board Agent", "static-site contract checks", "Keep default, challenger, and experimental outputs clearly separated."],
]
POSITION_TRUST_COLUMNS = ["pos_g", "years", "n", "mean_delta", "median_delta", "win_rate", "worst_delta", "trust_label"]
PICK_BUCKET_COLUMNS = ["pick_bucket", "n", "apex_spearman", "market_spearman", "lift"]
EDGE_CAL_COLUMNS = ["edge_band", "n", "beat_market_rate", "avg_actual_vs_market", "avg_edge"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"error": f"Could not parse {path.name}"}


def num(series) -> pd.Series:
    return pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def delta_summary(series) -> dict[str, Any]:
    clean = num(series)
    if clean.empty:
        return {"n": 0, "mean": None, "median": None, "win_rate": None, "worst": None, "best": None}
    return {
        "n": int(len(clean)),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "win_rate": round(float((clean > 0).mean()), 4),
        "worst": round(float(clean.min()), 4),
        "best": round(float(clean.max()), 4),
    }


def yearly_model(df: pd.DataFrame, delta_col: str) -> dict[str, Any]:
    if df.empty or delta_col not in df.columns:
        return {"status": "missing", "summary": delta_summary([])}
    out = {"status": "available", "summary": delta_summary(df[delta_col])}
    if "test_year" in df.columns:
        tmp = df[["test_year", delta_col]].copy()
        tmp[delta_col] = pd.to_numeric(tmp[delta_col], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty:
            best = tmp.loc[tmp[delta_col].idxmax()]
            worst = tmp.loc[tmp[delta_col].idxmin()]
            out["best_year"] = {"year": int(best.test_year), "delta": round(float(best[delta_col]), 4)}
            out["worst_year"] = {"year": int(worst.test_year), "delta": round(float(worst[delta_col]), 4)}
    return out


def trust(mean: float | None, worst: float | None, win: float | None, years: int) -> str:
    if mean is None or worst is None or win is None or years < 3:
        return "insufficient_sample"
    if mean > 0.02 and worst > -0.02 and win >= 0.70:
        return "candidate_for_more_trust"
    if mean > 0 and win >= 0.60:
        return "track_but_do_not_promote"
    if mean < 0 or worst < -0.05:
        return "shrink_or_suppress"
    return "neutral_watch"


def position_trust(pos: pd.DataFrame) -> pd.DataFrame:
    if pos.empty or "pos_g" not in pos.columns or "delta_raw_vs_pick" not in pos.columns:
        return pd.DataFrame(columns=POSITION_TRUST_COLUMNS)
    rows = []
    for p, g in pos.groupby("pos_g"):
        d = num(g["delta_raw_vs_pick"])
        if d.empty:
            continue
        n = int(pd.to_numeric(g.get("n", pd.Series([0] * len(g))), errors="coerce").fillna(0).sum())
        row = {
            "pos_g": str(p),
            "years": int(len(d)),
            "n": n,
            "mean_delta": round(float(d.mean()), 4),
            "median_delta": round(float(d.median()), 4),
            "win_rate": round(float((d > 0).mean()), 4),
            "worst_delta": round(float(d.min()), 4),
        }
        row["trust_label"] = trust(row["mean_delta"], row["worst_delta"], row["win_rate"], row["years"])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=POSITION_TRUST_COLUMNS)
    return pd.DataFrame(rows, columns=POSITION_TRUST_COLUMNS).sort_values(["trust_label", "mean_delta"], ascending=[True, False])


def pick_score_cols(board: pd.DataFrame) -> tuple[str | None, str | None]:
    score = next((c for c in ["apex_conservative_050", "recommended_candidate_score", "apex_score", "apex", "apex_raw"] if c in board.columns), None)
    market = next((c for c in ["exp_at_pick", "market", "pick_only"] if c in board.columns), None)
    return score, market


def rho(a: pd.Series, b: pd.Series) -> float | None:
    f = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(f) < 5 or f.a.nunique() < 2 or f.b.nunique() < 2:
        return None
    return round(float(f.a.corr(f.b, method="spearman")), 4)


def pick_bucket_lift(board: pd.DataFrame) -> pd.DataFrame:
    score, market = pick_score_cols(board)
    if board.empty or not score or not market or "Pick" not in board.columns or "y" not in board.columns:
        return pd.DataFrame(columns=PICK_BUCKET_COLUMNS)
    df = board.copy()
    df["Pick"] = pd.to_numeric(df["Pick"], errors="coerce")
    df = df[df["Pick"].lt(263) & df["y"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=PICK_BUCKET_COLUMNS)
    df["pick_bucket"] = pd.cut(df.Pick, [0, 32, 64, 100, 150, 262], labels=["1-32", "33-64", "65-100", "101-150", "151-262"])
    rows = []
    for bucket, g in df.groupby("pick_bucket", observed=True):
        ar, mr = rho(g[score], g.y), rho(g[market], g.y)
        rows.append({"pick_bucket": str(bucket), "n": int(len(g)), "apex_spearman": ar, "market_spearman": mr, "lift": None if ar is None or mr is None else round(ar - mr, 4)})
    return pd.DataFrame(rows, columns=PICK_BUCKET_COLUMNS)


def edge_calibration(board: pd.DataFrame) -> pd.DataFrame:
    score, market = pick_score_cols(board)
    if board.empty or not score or not market or "y" not in board.columns:
        return pd.DataFrame(columns=EDGE_CAL_COLUMNS)
    df = board.copy()
    df["Pick"] = pd.to_numeric(df.get("Pick", np.nan), errors="coerce")
    df = df[df.Pick.lt(263) & df.y.notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=EDGE_CAL_COLUMNS)
    scale = 100.0 if pd.to_numeric(df[[score, market, "y"]].stack(), errors="coerce").median() > 1.5 else 1.0
    df["edge"] = (pd.to_numeric(df[score], errors="coerce") - pd.to_numeric(df[market], errors="coerce")) * (100.0 / scale)
    df["actual_vs_market"] = (pd.to_numeric(df.y, errors="coerce") - pd.to_numeric(df[market], errors="coerce")) * (100.0 / scale)
    df = df.dropna(subset=["edge", "actual_vs_market"])
    if df.empty:
        return pd.DataFrame(columns=EDGE_CAL_COLUMNS)
    df["edge_band"] = pd.cut(df.edge, [-np.inf, -5, -2, 2, 5, np.inf], labels=["strong_fade", "fade", "neutral", "value", "strong_value"])
    rows = []
    for band, g in df.groupby("edge_band", observed=True):
        rows.append({"edge_band": str(band), "n": int(len(g)), "beat_market_rate": round(float((g.actual_vs_market > 0).mean()), 4), "avg_actual_vs_market": round(float(g.actual_vs_market.mean()), 2), "avg_edge": round(float(g.edge.mean()), 2)})
    return pd.DataFrame(rows, columns=EDGE_CAL_COLUMNS)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def work_queue(report: dict[str, Any], pos: pd.DataFrame, buckets: pd.DataFrame) -> list[dict[str, str]]:
    tasks = []
    current = report["yearly_models"]["profile_raw_apex"]["summary"]
    if current.get("worst") is not None and current["worst"] < -0.02:
        tasks.append({"priority": "P0", "owner_agent": "Validation Agent", "task": "Reduce worst-year drawdown before promoting any stronger score.", "gate": "Worst-year delta >= -0.020 while mean lift stays positive."})
    weak_pos = pos[pos.trust_label.eq("shrink_or_suppress")].pos_g.tolist() if not pos.empty and "trust_label" in pos.columns else []
    if weak_pos:
        tasks.append({"priority": "P0", "owner_agent": "Position Specialist Agent", "task": f"Add trust-aware residual labels/gates for unstable positions: {', '.join(weak_pos)}.", "gate": "Negative position edges are shrunk or labeled scout-required."})
    weak_buckets = []
    if not buckets.empty and "lift" in buckets.columns:
        weak_buckets = buckets[pd.to_numeric(buckets["lift"], errors="coerce").fillna(0).lt(0)].pick_bucket.tolist()
    if weak_buckets:
        tasks.append({"priority": "P1", "owner_agent": "Calibration Agent", "task": f"Calibrate or suppress edges in weak pick ranges: {', '.join(weak_buckets)}.", "gate": "Bucket lift non-negative or visibly low-confidence."})
    tasks.append({"priority": "P1", "owner_agent": "Data QA Agent", "task": "Block zero-row dashboard publishes.", "gate": "GitHub Pages cannot publish an empty board."})
    tasks.append({"priority": "P2", "owner_agent": "Product/Board Agent", "task": "Expose position and pick-range trust labels before changing scores.", "gate": "Improves decision hygiene without score promotion."})
    return tasks


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = ["# Zero-cost APEX forecasting sprint", "", "## Agent team", "", "| Agent | Engine | Assignment |", "|---|---|---|"]
    lines += [f"| {a[0]} | {a[1]} | {a[2]} |" for a in AGENTS]
    lines += ["", "## Current checks", ""]
    for name, item in report["yearly_models"].items():
        s = item["summary"]
        lines.append(f"- **{name}**: status={item['status']}; mean={s.get('mean')}; median={s.get('median')}; win_rate={s.get('win_rate')}; worst={s.get('worst')}")
    lines += ["", "## Work queue", ""]
    for t in report["work_queue"]:
        lines.append(f"- **{t['priority']} / {t['owner_agent']}** — {t['task']} Gate: {t['gate']}")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()
    reports, data, out = Path(args.reports_dir), Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolling = read_csv(reports / "rolling_backtest_summary.csv")
    pos_raw = read_csv(reports / "rolling_backtest_by_position.csv")
    nested = read_csv(reports / "nested_factor_backtest_summary.csv")
    ensemble = read_csv(reports / "position_ensemble_backtest_summary.csv")
    board = read_csv(data / "apex_board.csv")
    pos = position_trust(pos_raw)
    buckets = pick_bucket_lift(board)
    edge = edge_calibration(board)
    report = {
        "agent_roster": [{"agent": a, "engine": b, "assignment": c, "cost": "$0"} for a, b, c in AGENTS],
        "inputs": {"reports_dir": str(reports), "data_dir": str(data), "rolling_rows": len(rolling), "position_rows": len(pos_raw), "board_rows": len(board), "uses_paid_data_or_apis": False},
        "yearly_models": {
            "profile_raw_apex": yearly_model(rolling, "delta_raw_vs_pick_spearman_drafted"),
            "nested_factor": yearly_model(nested, "delta_nested_vs_pick_spearman_drafted"),
            "position_ensemble": yearly_model(ensemble, "delta_ensemble_vs_pick_spearman_drafted"),
        },
        "predraft_status": read_json(reports / "predraft_backtest_report.json"),
        "feature_ablation_status": records(read_csv(reports / "feature_ablation_summary.csv")),
        "position_trust": records(pos),
        "pick_bucket_lift": records(buckets),
        "edge_calibration": records(edge),
    }
    report["work_queue"] = work_queue(report, pos, buckets)
    (out / "forecasting_sprint_report.json").write_text(json.dumps(report, indent=2))
    pos.to_csv(out / "forecasting_position_trust.csv", index=False)
    buckets.to_csv(out / "forecasting_pick_bucket_lift.csv", index=False)
    edge.to_csv(out / "forecasting_edge_calibration.csv", index=False)
    write_md(report, out / "forecasting_agent_tasks.md")
    print(json.dumps({"board_rows": len(board), "tasks": len(report["work_queue"]), "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
