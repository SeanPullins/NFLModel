"""Audit optional CFBD production features before any model promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, POSMAP, norm

CFBD_PATH = ROOT / "data" / "production" / "cfbd_production.csv"
DRAFT_PATH = ROOT / "data" / "draft_data.csv"
SUMMARY_PATH = ROOT / "reports" / "cfbd_signal_audit.csv"
REPORT_PATH = ROOT / "reports" / "cfbd_signal_audit.json"
RECENT_PATH = ROOT / "reports" / "cfbd_recent_pick_notes.csv"
ERAS = [(2004, 2009, "2004-2009"), (2010, 2015, "2010-2015"), (2016, 2021, "2016-2021")]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def make_key(df: pd.DataFrame) -> pd.Series:
    return df["Player"].map(norm) + "_" + pd.to_numeric(df["Year"], errors="coerce").astype("Int64").astype(str)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("cfbd_")]


def spearman(a: pd.Series, b: pd.Series, min_n: int = 20) -> float | None:
    f = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(f) < min_n or f.a.nunique() < 2 or f.b.nunique() < 2:
        return None
    return float(f.a.corr(f.b, method="spearman"))


def residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    f = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    out = pd.Series(np.nan, index=y.index, dtype=float)
    if len(f) < 20 or f.x.nunique() < 2:
        return out
    mat = np.column_stack([np.ones(len(f)), f.x.to_numpy(float)])
    beta, *_ = np.linalg.lstsq(mat, f.y.to_numpy(float), rcond=None)
    out.loc[f.index] = f.y - mat.dot(beta)
    return out


def merge_data(draft: pd.DataFrame, cfbd: pd.DataFrame) -> pd.DataFrame:
    if draft.empty or cfbd.empty:
        return pd.DataFrame()
    d = draft.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d = d[d.Year.notna() & d.Player.notna()].copy()
    d["Year"] = d["Year"].astype(int)
    d["Pick"] = pd.to_numeric(d.get("Pick"), errors="coerce")
    d["CarAV"] = pd.to_numeric(d.get("CarAV"), errors="coerce")
    d["pos_g"] = d.get("Pos", "").map(POSMAP).fillna("OTH") if "Pos" in d.columns else "OTH"
    d["key"] = make_key(d)
    d["y"] = d.groupby("Year")["CarAV"].rank(pct=True)
    d.loc[d.groupby("Year")["CarAV"].transform("max").eq(0), "y"] = np.nan

    c = cfbd.copy()
    c["Year"] = pd.to_numeric(c["Year"], errors="coerce")
    c = c[c.Year.notna() & c.Player.notna()].copy()
    c["Year"] = c["Year"].astype(int)
    c["key"] = make_key(c)
    return d.merge(c[["key", *feature_cols(c)]].drop_duplicates("key"), on="key", how="left")


def tier(residual_rho: float | None, stability: float | None, n: int) -> str:
    if residual_rho is None or stability is None or n < 80:
        return "insufficient"
    ar = abs(residual_rho)
    if ar >= 0.06 and stability >= 0.67:
        return "A_candidate"
    if ar >= 0.035 and stability >= 0.67:
        return "B_watch"
    if ar >= 0.02:
        return "C_note"
    return "reject"


def audit_one(df: pd.DataFrame, feature: str, min_n: int) -> dict:
    s = df[df.y.notna() & df.Pick.lt(263) & df[feature].notna()].copy()
    n = int(len(s))
    row = {"feature": feature, "n": n}
    if n < min_n:
        row.update({"raw_spearman": None, "residual_spearman": None, "era_stability": None, "tier": "insufficient"})
        return row

    logpick = np.log(s.Pick.clip(1, 263))
    raw = spearman(s[feature], s.y)
    fr = residualize(s[feature], logpick)
    yr = residualize(s.y, logpick)
    resid = spearman(fr, yr)
    era_vals = []
    for lo, hi, _label in ERAS:
        g = s[s.Year.between(lo, hi)]
        if len(g) < max(25, min_n // 4):
            continue
        gf = residualize(g[feature], np.log(g.Pick.clip(1, 263)))
        gy = residualize(g.y, np.log(g.Pick.clip(1, 263)))
        val = spearman(gf, gy)
        if val is not None and np.isfinite(val):
            era_vals.append(float(val))
    stability = None
    if resid is not None and era_vals:
        sign = 1 if resid >= 0 else -1
        stability = float(np.mean([np.sign(v) == sign for v in era_vals]))
    row.update(
        {
            "raw_spearman": None if raw is None else round(raw, 4),
            "residual_spearman": None if resid is None else round(resid, 4),
            "era_stability": None if stability is None else round(stability, 4),
            "era_values": [round(v, 4) for v in era_vals],
        }
    )
    row["tier"] = tier(row["residual_spearman"], row["era_stability"], n)
    return row


def by_position(df: pd.DataFrame, features: list[str], min_n: int) -> pd.DataFrame:
    rows = []
    for pos, g in df.groupby("pos_g", dropna=False):
        for feat in features:
            row = audit_one(g, feat, max(30, min_n // 2))
            if row["tier"] != "insufficient":
                row["pos_g"] = str(pos)
                rows.append(row)
    return pd.DataFrame(rows)


def recent_notes(df: pd.DataFrame, summary: pd.DataFrame, recent_start: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    use = summary[summary.tier.isin(["A_candidate", "B_watch"])].feature.tolist()
    if not use:
        use = summary.sort_values("residual_spearman", key=lambda s: s.abs(), ascending=False).head(8).feature.tolist()
    use = [f for f in use if f in df.columns]
    if not use:
        return pd.DataFrame()
    hist = df[df.Year < recent_start]
    recent = df[df.Year >= recent_start].copy()
    keep = ["Year", "Player", "Pos", "pos_g", "College", "Pick", *use]
    out = recent[[c for c in keep if c in recent.columns]].copy()
    for feat in use:
        vals = pd.to_numeric(hist[feat], errors="coerce")
        mu, sd = vals.mean(), vals.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[f"{feat}_hist_z"] = ((pd.to_numeric(out[feat], errors="coerce") - mu) / sd).clip(-4, 4)
    zcols = [c for c in out.columns if c.endswith("_hist_z")]
    out["cfbd_positive_note_count"] = out[zcols].gt(1).sum(axis=1)
    out["cfbd_caution_note_count"] = out[zcols].lt(-1).sum(axis=1)
    return out.sort_values(["Year", "cfbd_positive_note_count", "Pick"], ascending=[False, False, True])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default=str(DRAFT_PATH))
    p.add_argument("--cfbd", default=str(CFBD_PATH))
    p.add_argument("--summary", default=str(SUMMARY_PATH))
    p.add_argument("--report", default=str(REPORT_PATH))
    p.add_argument("--recent-notes", default=str(RECENT_PATH))
    p.add_argument("--min-n", type=int, default=80)
    p.add_argument("--recent-start", type=int, default=2022)
    args = p.parse_args()

    df = merge_data(read_csv(Path(args.draft)), read_csv(Path(args.cfbd)))
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        report = {"status": "skipped_missing_data", "message": "Run src/build_cfbd_production.py first.", "uses_paid_data_or_apis": False}
        pd.DataFrame(columns=["feature", "n", "raw_spearman", "residual_spearman", "era_stability", "tier"]).to_csv(args.summary, index=False)
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    feats = feature_cols(df)
    summary = pd.DataFrame([audit_one(df, f, args.min_n) for f in feats]).sort_values(["tier", "residual_spearman"], ascending=[True, False])
    pos = by_position(df, feats, args.min_n)
    notes = recent_notes(df, summary, args.recent_start)
    summary.to_csv(args.summary, index=False)
    Path(args.summary).with_name("cfbd_signal_audit_by_position.csv").write_text(pos.to_csv(index=False))
    notes.to_csv(args.recent_notes, index=False)
    report = {
        "status": "ok",
        "rows": int(len(df)),
        "features_tested": int(len(feats)),
        "top_candidates": summary.head(12).replace({np.nan: None}).to_dict(orient="records"),
        "recent_notes_rows": int(len(notes)),
        "promotion_contract": "Research only until a rolling out-of-time model with CFBD features passes APEX promotion gates.",
        "uses_paid_data_or_apis": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": "ok", "features_tested": len(feats), "recent_notes_rows": len(notes)}, indent=2))


if __name__ == "__main__":
    main()
