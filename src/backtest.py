"""Rolling out-of-time validation for APEX.

Examples:
    python src/backtest.py
    APEX_DATA_DIR=/path/to/raw python src/backtest.py --first-test-year 2011 --last-test-year 2016
    python src/backtest.py --apex-plus-factor 3.5

The key output is the paired lift of APEX+ versus the pick-only market baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline import (
    ROOT,
    load_dataset,
    make_baseline,
    make_pick_baseline,
    make_resid,
    metric_row,
    prepare_fold,
    safe_spearman,
    score_apex,
    tune_position_shrinkage,
)

DEFAULT_APEX_PLUS_FACTOR = 3.5


def bootstrap_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 7) -> dict[str, float]:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(clean) == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    if len(clean) == 1:
        value = float(clean[0])
        return {"mean": value, "lo": value, "hi": value, "n": 1}

    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(n_boot, len(clean)), replace=True).mean(axis=1)
    return {
        "mean": float(clean.mean()),
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "n": int(len(clean)),
    }


def paired_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "win_rate": np.nan,
            "n": 0,
        }
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "win_rate": float((clean > 0).mean()),
        "n": int(len(clean)),
    }


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def score_apex_plus(
    market: np.ndarray,
    apex_raw: np.ndarray,
    factor: float = DEFAULT_APEX_PLUS_FACTOR,
    lower: float = 0.01,
    upper: float = 0.99,
) -> np.ndarray:
    """Amplify the residual disagreement between raw APEX and the market baseline."""
    return np.clip(market + factor * (apex_raw - market), lower, upper)


def evaluate_test_year(
    df: pd.DataFrame,
    test_year: int,
    validation_years: int = 2,
    apex_plus_factor: float = DEFAULT_APEX_PLUS_FACTOR,
) -> tuple[dict, list[dict]]:
    """Evaluate one test year with all transforms fit only on prior years."""
    train_for_shrink_end = test_year - validation_years - 1
    valid_start = test_year - validation_years
    valid_end = test_year - 1

    train_for_shrink_raw = df[df["Year"] <= train_for_shrink_end].copy()
    valid_raw = df[(df["Year"] >= valid_start) & (df["Year"] <= valid_end)].copy()
    final_train_raw = df[df["Year"] < test_year].copy()
    test_raw = df[df["Year"] == test_year].copy()

    if train_for_shrink_raw.empty or valid_raw.empty or final_train_raw.empty or test_raw.empty:
        raise ValueError(f"Not enough data to evaluate {test_year}")

    train_for_shrink, valid = prepare_fold(train_for_shrink_raw, valid_raw)
    base_for_shrink, _, _ = make_baseline(train_for_shrink)
    resid_for_shrink, _ = make_resid(train_for_shrink, base_for_shrink)
    shrink = tune_position_shrinkage(valid, base_for_shrink, resid_for_shrink)

    final_train, test = prepare_fold(final_train_raw, test_raw)
    pick_only, _ = make_pick_baseline(final_train)
    base, _, _ = make_baseline(final_train)
    resid, _ = make_resid(final_train, base)

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["market"] = base(scored)
    scored["apex_raw"] = score_apex(scored, base, resid, shrink)
    scored["apex_plus"] = score_apex_plus(scored["market"].to_numpy(), scored["apex_raw"].to_numpy(), apex_plus_factor)

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "shrink_train_years": f"<= {train_for_shrink_end}",
        "validation_years": f"{valid_start}-{valid_end}",
        "apex_plus_factor": apex_plus_factor,
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
    }
    row.update(flatten_metrics("pick_only", metric_row(scored, "pick_only")))
    row.update(flatten_metrics("market", metric_row(scored, "market")))
    row.update(flatten_metrics("apex_raw", metric_row(scored, "apex_raw")))
    row.update(flatten_metrics("apex_plus", metric_row(scored, "apex_plus")))
    row["delta_raw_vs_pick_spearman_drafted"] = row["apex_raw_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_plus_vs_pick_spearman_drafted"] = row["apex_plus_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_plus_vs_raw_spearman_drafted"] = row["apex_plus_spearman_drafted"] - row["apex_raw_spearman_drafted"]
    row["delta_plus_vs_market_spearman_drafted"] = row["apex_plus_spearman_drafted"] - row["market_spearman_drafted"]

    pos_rows: list[dict] = []
    drafted = scored[scored["Pick"].lt(263)].copy()
    for pos, group in drafted.groupby("pos_g", observed=True):
        if len(group) < 20:
            continue
        pos_rows.append(
            {
                "test_year": test_year,
                "pos_g": str(pos),
                "n": int(len(group)),
                "pick_only_spearman": safe_spearman(group["pick_only"], group["y"]),
                "market_spearman": safe_spearman(group["market"], group["y"]),
                "apex_raw_spearman": safe_spearman(group["apex_raw"], group["y"]),
                "apex_plus_spearman": safe_spearman(group["apex_plus"], group["y"]),
                "delta_plus_vs_pick": safe_spearman(group["apex_plus"], group["y"]) - safe_spearman(group["pick_only"], group["y"]),
                "shrink": shrink.get(str(pos), 0.4),
            }
        )

    return row, pos_rows


def aggregate_report(summary: pd.DataFrame, first_test_year: int, last_test_year: int, validation_years: int, apex_plus_factor: float) -> dict:
    deltas_plus = summary["delta_plus_vs_pick_spearman_drafted"].tolist() if not summary.empty else []
    deltas_raw = summary["delta_raw_vs_pick_spearman_drafted"].tolist() if not summary.empty else []
    deltas_plus_vs_raw = summary["delta_plus_vs_raw_spearman_drafted"].tolist() if not summary.empty else []

    if summary.empty:
        best_year = worst_year = None
    else:
        best = summary.loc[summary["delta_plus_vs_pick_spearman_drafted"].idxmax()]
        worst = summary.loc[summary["delta_plus_vs_pick_spearman_drafted"].idxmin()]
        best_year = {
            "test_year": int(best["test_year"]),
            "delta_plus_vs_pick_spearman_drafted": float(best["delta_plus_vs_pick_spearman_drafted"]),
        }
        worst_year = {
            "test_year": int(worst["test_year"]),
            "delta_plus_vs_pick_spearman_drafted": float(worst["delta_plus_vs_pick_spearman_drafted"]),
        }

    return {
        "first_test_year": first_test_year,
        "last_test_year": last_test_year,
        "validation_years": validation_years,
        "years_evaluated": int(len(summary)),
        "apex_plus_factor": apex_plus_factor,
        "primary_metric": "delta_plus_vs_pick_spearman_drafted",
        "apex_plus_vs_pick": {
            **paired_summary(summary.get("delta_plus_vs_pick_spearman_drafted", pd.Series(dtype=float))),
            "bootstrap_ci": bootstrap_ci(deltas_plus),
        },
        "apex_raw_vs_pick": {
            **paired_summary(summary.get("delta_raw_vs_pick_spearman_drafted", pd.Series(dtype=float))),
            "bootstrap_ci": bootstrap_ci(deltas_raw),
        },
        "apex_plus_vs_raw": {
            **paired_summary(summary.get("delta_plus_vs_raw_spearman_drafted", pd.Series(dtype=float))),
            "bootstrap_ci": bootstrap_ci(deltas_plus_vs_raw),
        },
        "best_year": best_year,
        "worst_year": worst_year,
    }


def run_backtest(
    first_test_year: int,
    last_test_year: int,
    validation_years: int,
    data_dir: str | None,
    apex_plus_factor: float = DEFAULT_APEX_PLUS_FACTOR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_dataset(data_dir=data_dir)

    rows: list[dict] = []
    pos_rows: list[dict] = []
    for test_year in range(first_test_year, last_test_year + 1):
        try:
            row, year_pos_rows = evaluate_test_year(df, test_year, validation_years, apex_plus_factor)
        except ValueError as exc:
            print(f"Skipping {test_year}: {exc}")
            continue
        rows.append(row)
        pos_rows.extend(year_pos_rows)
        print(
            f"{test_year}: APEX+ drafted Spearman={row['apex_plus_spearman_drafted']:.3f} "
            f"vs pick={row['pick_only_spearman_drafted']:.3f} "
            f"delta={row['delta_plus_vs_pick_spearman_drafted']:.3f}"
        )

    summary = pd.DataFrame(rows)
    pos_summary = pd.DataFrame(pos_rows)
    report = aggregate_report(summary, first_test_year, last_test_year, validation_years, apex_plus_factor)
    return summary, pos_summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2016)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--apex-plus-factor", type=float, default=DEFAULT_APEX_PLUS_FACTOR)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    summary, pos_summary, report = run_backtest(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        validation_years=args.validation_years,
        data_dir=args.data_dir,
        apex_plus_factor=args.apex_plus_factor,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "rolling_backtest_summary.csv", index=False)
    pos_summary.round(4).to_csv(out_dir / "rolling_backtest_by_position.csv", index=False)
    (out_dir / "rolling_backtest_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Wrote rolling backtest outputs to {out_dir}")


if __name__ == "__main__":
    main()
