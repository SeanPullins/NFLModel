"""Rolling out-of-time validation for APEX.

Examples:
    python src/backtest.py
    APEX_DATA_DIR=/path/to/raw python src/backtest.py --first-test-year 2011 --last-test-year 2016
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 7) -> dict[str, float]:
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


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_test_year(df: pd.DataFrame, test_year: int, validation_years: int = 2) -> tuple[dict, list[dict]]:
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
    scored["pos_base"] = base(scored)
    scored["apex"] = score_apex(scored, base, resid, shrink)

    row = {
        "test_year": test_year,
        "train_years": f"<= {test_year - 1}",
        "shrink_train_years": f"<= {train_for_shrink_end}",
        "validation_years": f"{valid_start}-{valid_end}",
        "n_all": int(len(scored)),
        "n_drafted": int(scored["Pick"].lt(263).sum()),
    }
    row.update(flatten_metrics("pick_only", metric_row(scored, "pick_only")))
    row.update(flatten_metrics("pos_base", metric_row(scored, "pos_base")))
    row.update(flatten_metrics("apex", metric_row(scored, "apex")))
    row["delta_apex_vs_pick_spearman_drafted"] = row["apex_spearman_drafted"] - row["pick_only_spearman_drafted"]
    row["delta_apex_vs_pos_base_spearman_drafted"] = row["apex_spearman_drafted"] - row["pos_base_spearman_drafted"]

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
                "pos_base_spearman": safe_spearman(group["pos_base"], group["y"]),
                "apex_spearman": safe_spearman(group["apex"], group["y"]),
                "shrink": shrink.get(str(pos), 0.4),
            }
        )

    return row, pos_rows


def run_backtest(
    first_test_year: int,
    last_test_year: int,
    validation_years: int,
    data_dir: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_dataset(data_dir=data_dir)

    rows: list[dict] = []
    pos_rows: list[dict] = []
    for test_year in range(first_test_year, last_test_year + 1):
        try:
            row, year_pos_rows = evaluate_test_year(df, test_year, validation_years)
        except ValueError as exc:
            print(f"Skipping {test_year}: {exc}")
            continue
        rows.append(row)
        pos_rows.extend(year_pos_rows)
        print(
            f"{test_year}: APEX drafted Spearman={row['apex_spearman_drafted']:.3f} "
            f"vs pick={row['pick_only_spearman_drafted']:.3f} "
            f"delta={row['delta_apex_vs_pick_spearman_drafted']:.3f}"
        )

    summary = pd.DataFrame(rows)
    pos_summary = pd.DataFrame(pos_rows)
    deltas = summary["delta_apex_vs_pick_spearman_drafted"].tolist() if not summary.empty else []
    report = {
        "first_test_year": first_test_year,
        "last_test_year": last_test_year,
        "validation_years": validation_years,
        "years_evaluated": int(len(summary)),
        "apex_vs_pick_spearman_drafted_delta_ci": bootstrap_ci(deltas),
    }
    return summary, pos_summary, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2016)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    summary, pos_summary, report = run_backtest(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        validation_years=args.validation_years,
        data_dir=args.data_dir,
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
