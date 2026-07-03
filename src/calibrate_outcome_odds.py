"""Create display-friendly outcome odds and exact slot-value fields.

This does not promote a new forecasting model. It creates a dashboard display
layer so bust/star odds are more useful than broad bucket rates and so the site
can show exact Actual Pick, Model Pick, and Slot Value.

Fitting/evaluation uses mature classes only (2011-2021 by default). Recent
classes never tune the curves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOARD_PATH = ROOT / "data" / "apex_board.csv"
REPORT_PATH = ROOT / "reports" / "outcome_odds_calibration_report.json"
CURVE_PATH = ROOT / "reports" / "outcome_odds_calibration.csv"
MATURE_START = 2011
MATURE_END = 2021
MAX_PICK = 262


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def weighted_slot_curve(picks: pd.Series, outcome: pd.Series, slots: np.ndarray, increasing: bool) -> pd.Series:
    data = pd.DataFrame({"pick": pd.to_numeric(picks, errors="coerce"), "outcome": pd.to_numeric(outcome, errors="coerce")}).dropna()
    if data.empty:
        return pd.Series(0.0, index=slots)
    values: list[float] = []
    for slot in slots:
        # Wider windows later in the draft keep the curve stable without making
        # top-10 picks look identical to late-round picks.
        window = max(10, min(42, int(round(slot * 0.16))))
        sample = data[data["pick"].between(slot - window, slot + window)].copy()
        if len(sample) < 25:
            sample = data.iloc[(data["pick"] - slot).abs().argsort()[: min(len(data), 70)]].copy()
        dist = (sample["pick"] - slot).abs()
        weights = 1.0 / (1.0 + dist)
        values.append(float(np.average(sample["outcome"], weights=weights)))
    curve = pd.Series(values, index=slots).clip(0.01, 0.99)
    # Draft slot should be a broad prior: miss risk generally rises as the pick
    # gets later, while star/starter chances generally fall. Enforce that shape
    # gently to avoid noisy local reversals.
    if increasing:
        curve = curve.cummax()
    else:
        curve = curve[::-1].cummax()[::-1]
    return curve.clip(0.01, 0.99)


def interp_curve(curve: pd.Series, slots: pd.Series) -> pd.Series:
    x = pd.to_numeric(slots, errors="coerce").clip(1, MAX_PICK)
    out = pd.Series(np.nan, index=slots.index, dtype="float64")
    good = x.notna()
    if good.any():
        out.loc[good] = np.interp(x.loc[good].to_numpy(), curve.index.to_numpy(), curve.to_numpy())
    return out


def brier(pred: pd.Series, actual: pd.Series) -> float | None:
    data = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if len(data) < 50:
        return None
    return float(((data["pred"] - data["actual"]) ** 2).mean())


def ece(pred: pd.Series, actual: pd.Series) -> float | None:
    data = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if len(data) < 50:
        return None
    bins = pd.cut(data["pred"], np.linspace(0, 1, 11), include_lowest=True)
    rel = data.groupby(bins, observed=True).agg(pred=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    return float((rel["n"] / len(data) * (rel["pred"] - rel["actual"]).abs()).sum())


def slot_label(slot_value: pd.Series) -> pd.Series:
    out = pd.Series("No pick data", index=slot_value.index, dtype="object")
    value = pd.to_numeric(slot_value, errors="coerce")
    out[value.notna() & value.between(-3, 3)] = "Fair value"
    out[value.gt(3)] = value[value.gt(3)].round().astype(int).map(lambda x: f"Value +{x} slots")
    out[value.lt(-3)] = value[value.lt(-3)].round().astype(int).map(lambda x: f"Reach {x} slots")
    return out


def bust_band(p: pd.Series) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce")
    out = pd.Series("Unknown", index=p.index, dtype="object")
    out[x.notna() & x.lt(0.18)] = "Low"
    out[x.notna() & x.ge(0.18) & x.lt(0.32)] = "Medium"
    out[x.notna() & x.ge(0.32) & x.lt(0.48)] = "High"
    out[x.notna() & x.ge(0.48)] = "Very High"
    return out


def add_display_odds(board: pd.DataFrame, report_path: Path = REPORT_PATH, curve_path: Path = CURVE_PATH) -> pd.DataFrame:
    out = board.copy()
    year = num(out, "Year")
    pick = num(out, "Pick")
    model_pick = num(out, "implied_pick")
    model_pick = model_pick.where(model_pick.between(1, MAX_PICK))
    actual_pick = pick.where(pick.between(1, MAX_PICK))
    slot_value = actual_pick - model_pick

    mature = out[year.between(MATURE_START, MATURE_END) & num(out, "y").notna() & actual_pick.notna()].copy()
    mature_pick = pd.to_numeric(mature["Pick"], errors="coerce")
    mature_y = pd.to_numeric(mature["y"], errors="coerce")
    slots = np.arange(1, MAX_PICK + 1)

    outcomes = {
        "star": mature_y.ge(0.90).astype(float),
        "starter": mature_y.ge(0.60).astype(float),
        "role": mature_y.ge(0.45).astype(float),
        "bust": mature_y.le(0.30).astype(float),
    }
    curves = {
        "star": weighted_slot_curve(mature_pick, outcomes["star"], slots, increasing=False),
        "starter": weighted_slot_curve(mature_pick, outcomes["starter"], slots, increasing=False),
        "role": weighted_slot_curve(mature_pick, outcomes["role"], slots, increasing=False),
        "bust": weighted_slot_curve(mature_pick, outcomes["bust"], slots, increasing=True),
    }

    # Blend actual draft slot with model fair slot for display odds. This keeps
    # the market prior dominant while letting APEX edge create visible separation.
    adjusted_slot = actual_pick.copy()
    has_both = actual_pick.notna() & model_pick.notna()
    adjusted_slot.loc[has_both] = (0.70 * actual_pick.loc[has_both] + 0.30 * model_pick.loc[has_both]).clip(1, MAX_PICK)

    out["display_actual_pick"] = actual_pick
    out["display_model_pick"] = model_pick
    out["display_slot_value"] = slot_value
    out["display_slot_label"] = slot_label(slot_value)
    out["display_star_pct"] = interp_curve(curves["star"], adjusted_slot)
    out["display_starter_pct"] = interp_curve(curves["starter"], adjusted_slot)
    out["display_role_pct"] = interp_curve(curves["role"], adjusted_slot)
    out["display_bust_pct"] = interp_curve(curves["bust"], adjusted_slot)
    out["display_bust_band"] = bust_band(out["display_bust_pct"])
    out["odds_calibration_note"] = "Slot-calibrated display odds; model remains conservative/profile APEX"

    # Fallback for undrafted / missing pick rows.
    for dst, src in [
        ("display_star_pct", "p_star"),
        ("display_starter_pct", "p_starter"),
        ("display_role_pct", "p_contrib"),
        ("display_bust_pct", "p_bust"),
    ]:
        if src in out.columns:
            out[dst] = pd.to_numeric(out[dst], errors="coerce").fillna(pd.to_numeric(out[src], errors="coerce"))
    out["display_bust_band"] = bust_band(out["display_bust_pct"])

    # Report calibration on mature classes for old vs new display probabilities.
    mature_idx = mature.index
    actual_bust = outcomes["bust"]
    old_bust = pd.to_numeric(out.loc[mature_idx, "p_bust"], errors="coerce") if "p_bust" in out.columns else pd.Series(np.nan, index=mature_idx)
    new_bust = pd.to_numeric(out.loc[mature_idx, "display_bust_pct"], errors="coerce")
    actual_star = outcomes["star"]
    old_star = pd.to_numeric(out.loc[mature_idx, "p_star"], errors="coerce") if "p_star" in out.columns else pd.Series(np.nan, index=mature_idx)
    new_star = pd.to_numeric(out.loc[mature_idx, "display_star_pct"], errors="coerce")

    curve_table = pd.DataFrame({
        "pick": slots,
        "slot_star_pct": curves["star"].values,
        "slot_starter_pct": curves["starter"].values,
        "slot_role_pct": curves["role"].values,
        "slot_bust_pct": curves["bust"].values,
    })
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_table.round(4).to_csv(curve_path, index=False)

    report = {
        "mature_years": [MATURE_START, MATURE_END],
        "rows": int(len(out)),
        "mature_rows": int(len(mature)),
        "method": "70% actual pick slot + 30% model fair slot, smoothed mature-class slot curves",
        "model_promotion": "none; display calibration only",
        "bust": {
            "old_brier": None if brier(old_bust, actual_bust) is None else round(brier(old_bust, actual_bust), 4),
            "new_brier": None if brier(new_bust, actual_bust) is None else round(brier(new_bust, actual_bust), 4),
            "old_ece": None if ece(old_bust, actual_bust) is None else round(ece(old_bust, actual_bust), 4),
            "new_ece": None if ece(new_bust, actual_bust) is None else round(ece(new_bust, actual_bust), 4),
            "old_std": round(float(old_bust.std(skipna=True)), 4) if old_bust.notna().any() else None,
            "new_std": round(float(new_bust.std(skipna=True)), 4) if new_bust.notna().any() else None,
        },
        "star": {
            "old_brier": None if brier(old_star, actual_star) is None else round(brier(old_star, actual_star), 4),
            "new_brier": None if brier(new_star, actual_star) is None else round(brier(new_star, actual_star), 4),
            "old_ece": None if ece(old_star, actual_star) is None else round(ece(old_star, actual_star), 4),
            "new_ece": None if ece(new_star, actual_star) is None else round(ece(new_star, actual_star), 4),
            "old_std": round(float(old_star.std(skipna=True)), 4) if old_star.notna().any() else None,
            "new_std": round(float(new_star.std(skipna=True)), 4) if new_star.notna().any() else None,
        },
        "warning": "Display odds are calibrated outcome buckets, not exact player-specific probabilities.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default=str(BOARD_PATH))
    parser.add_argument("--out", default=None, help="Defaults to overwriting --board")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--curve", default=str(CURVE_PATH))
    args = parser.parse_args()

    board_path = Path(args.board)
    out_path = Path(args.out) if args.out else board_path
    board = pd.read_csv(board_path, low_memory=False)
    scored = add_display_odds(board, Path(args.report), Path(args.curve))
    scored.round(4).to_csv(out_path, index=False)
    print(f"wrote display odds to {out_path}")
    print(Path(args.report).read_text())


if __name__ == "__main__":
    main()
