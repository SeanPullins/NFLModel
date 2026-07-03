"""Create display-friendly outcome odds and exact should-have-gone slots.

This does not promote a new forecasting model. It creates a dashboard display
layer so the site can show:

- actual draft pick
- unique model pick / should-have-gone slot
- slot value or reach
- calmer slot-anchored bust risk

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


def model_score(df: pd.DataFrame) -> pd.Series:
    for col in ["apex_conservative_050", "front_office_score", "prospect_lens_score", "apex", "exp_at_pick"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(0.0, index=df.index, dtype="float64")


def unique_model_pick_by_year(df: pd.DataFrame, actual_pick: pd.Series) -> pd.Series:
    """Return a unique should-have-gone slot for each drafted row.

    The previous display used rounded implied_pick, which can duplicate slots.
    Draft boards cannot have two players at #8, so this is a true ranking within
    each draft year. Actual pick is only a tie-breaker, not the driver.
    """
    score = model_score(df)
    ranked = pd.DataFrame({
        "Year": num(df, "Year"),
        "score": score,
        "actual_pick": actual_pick,
        "player": df.get("Player", pd.Series("", index=df.index)).astype(str),
    }, index=df.index)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = ranked[ranked["Year"].notna() & ranked["actual_pick"].notna() & ranked["score"].notna()].copy()
    for _, group in valid.groupby("Year"):
        order = group.sort_values(["score", "actual_pick", "player"], ascending=[False, True, True]).index
        out.loc[order] = np.arange(1, len(order) + 1, dtype=float)
    return out.where(out.between(1, MAX_PICK))


def weighted_slot_curve(picks: pd.Series, outcome: pd.Series, slots: np.ndarray, increasing: bool) -> pd.Series:
    data = pd.DataFrame({"pick": pd.to_numeric(picks, errors="coerce"), "outcome": pd.to_numeric(outcome, errors="coerce")}).dropna()
    if data.empty:
        return pd.Series(0.0, index=slots)
    values: list[float] = []
    for slot in slots:
        window = max(12, min(52, int(round(slot * 0.18))))
        sample = data[data["pick"].between(slot - window, slot + window)].copy()
        if len(sample) < 35:
            sample = data.iloc[(data["pick"] - slot).abs().argsort()[: min(len(data), 90)]].copy()
        dist = (sample["pick"] - slot).abs()
        weights = 1.0 / (1.0 + dist)
        values.append(float(np.average(sample["outcome"], weights=weights)))
    curve = pd.Series(values, index=slots).clip(0.01, 0.99)
    curve = curve.cummax() if increasing else curve[::-1].cummax()[::-1]
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
    out[x.notna() & x.lt(0.16)] = "Low"
    out[x.notna() & x.ge(0.16) & x.lt(0.28)] = "Medium"
    out[x.notna() & x.ge(0.28)] = "High"
    return out


def add_display_odds(board: pd.DataFrame, report_path: Path = REPORT_PATH, curve_path: Path = CURVE_PATH) -> pd.DataFrame:
    out = board.copy()
    year = num(out, "Year")
    pick = num(out, "Pick")
    actual_pick = pick.where(pick.between(1, MAX_PICK))
    model_pick = unique_model_pick_by_year(out, actual_pick)
    slot_value = actual_pick - model_pick

    mature_mask = year.between(MATURE_START, MATURE_END) & num(out, "y").notna() & actual_pick.notna()
    mature_y = num(out, "y").loc[mature_mask]
    mature_actual_pick = actual_pick.loc[mature_mask]
    mature_model_pick = model_pick.loc[mature_mask]
    slots = np.arange(1, MAX_PICK + 1)

    outcomes = {
        "star": mature_y.ge(0.90).astype(float),
        "starter": mature_y.ge(0.60).astype(float),
        "role": mature_y.ge(0.45).astype(float),
        "bust": mature_y.le(0.30).astype(float),
    }
    # Slot curves are based on actual historical draft slots. We then apply them
    # to the unique model rank for the display board.
    curves = {
        "star": weighted_slot_curve(mature_actual_pick, outcomes["star"], slots, increasing=False),
        "starter": weighted_slot_curve(mature_actual_pick, outcomes["starter"], slots, increasing=False),
        "role": weighted_slot_curve(mature_actual_pick, outcomes["role"], slots, increasing=False),
        "bust": weighted_slot_curve(mature_actual_pick, outcomes["bust"], slots, increasing=True),
    }

    # Risk slot answers: if the model says he should have gone #8, use #8 as the
    # anchor. Actual draft slot gets only a tiny nudge so late slides still show
    # slightly more uncertainty without turning the display ridiculous.
    risk_slot = model_pick.copy()
    both = model_pick.notna() & actual_pick.notna()
    risk_slot.loc[both] = (0.90 * model_pick.loc[both] + 0.10 * actual_pick.loc[both]).clip(1, MAX_PICK)

    star = interp_curve(curves["star"], risk_slot)
    starter = interp_curve(curves["starter"], risk_slot)
    role = interp_curve(curves["role"], risk_slot)
    raw_bust = interp_curve(curves["bust"], risk_slot)
    bust_base_rate = float(outcomes["bust"].mean()) if len(outcomes["bust"]) else 0.30
    # Calm the display: this is a decision-board miss risk, not a fake exact
    # probability. Shrink extremes and cap the top so it does not look absurd.
    bust = (0.72 * raw_bust + 0.28 * bust_base_rate).clip(0.05, 0.42)

    out["display_actual_pick"] = actual_pick
    out["display_model_pick"] = model_pick
    out["display_slot_value"] = slot_value
    out["display_slot_label"] = slot_label(slot_value)
    out["display_star_pct"] = star
    out["display_starter_pct"] = starter
    out["display_role_pct"] = role
    out["display_bust_pct"] = bust
    out["display_bust_band"] = bust_band(out["display_bust_pct"])
    out["odds_calibration_note"] = "Should-have-gone slot ranking; bust risk is historical slot miss risk, not exact certainty"

    # Fallback for non-drafted / missing pick rows.
    for dst, src in [
        ("display_star_pct", "p_star"),
        ("display_starter_pct", "p_starter"),
        ("display_role_pct", "p_contrib"),
        ("display_bust_pct", "p_bust"),
    ]:
        if src in out.columns:
            out[dst] = pd.to_numeric(out[dst], errors="coerce").fillna(pd.to_numeric(out[src], errors="coerce"))
    out["display_bust_band"] = bust_band(out["display_bust_pct"])

    mature_idx = mature_y.index
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
        "slot_bust_pct_raw": curves["bust"].values,
    })
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_table.round(4).to_csv(curve_path, index=False)

    duplicate_model_slots = int(out[actual_pick.notna()].duplicated(["Year", "display_model_pick"]).sum())
    report = {
        "mature_years": [MATURE_START, MATURE_END],
        "rows": int(len(out)),
        "mature_rows": int(len(mature_y)),
        "method": "unique yearly model rank is the should-have-gone slot; risk is 90% model rank + 10% actual slot",
        "model_promotion": "none; display calibration only",
        "duplicate_model_slots": duplicate_model_slots,
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
        "warning": "Display odds are historical slot miss-risk estimates, not exact player-specific probabilities.",
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
