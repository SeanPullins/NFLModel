"""Bust-avoidance model and trait stress test.

Goal: for top-100 picks, predict the probability the player busts
(within-class career percentile < 0.45, the tier system's "bust" line) and
identify which pre-draft traits actually flag landmines.

Two outputs:
1. Rolling out-of-time backtest 2011-2021: does a trait-based bust model beat
   the pick-only bust base rate (AUC / top-decile flag precision)?
2. Trait table on all mature classes: bust rate by feature bucket vs the
   base rate, so the "traits that project bust avoidance" are explicit.

Usage:
    python src/bust_model.py --first-test-year 2011 --last-test-year 2021 --end-year 2021
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from feature_sets import model_features_for
from pipeline import (
    CATS,
    ROOT,
    load_dataset,
    prepare_fold,
)

BUST_THRESHOLD = 0.45
TOP_PICK_CUTOFF = 100

TRAIT_FEATURES = [
    "age",
    "dash_z",
    "speed_score_z",
    "explosion_z",
    "agility_z",
    "bmi_z",
    "weight_z",
    "height_z",
    "bench_z",
    "college_games_z",
    "col_enc",
    "log_consensus_rank_z",
    "espn_grade_z",
    "consensus_vs_pick_z",
]


def add_drafted_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """Within-class CarAV percentile among drafted players only.

    The pipeline's `y` ranks against the whole class including hundreds of
    zero-AV undrafted players, which makes almost every top-100 pick look
    fine. Bust must be judged against drafted peers.
    """
    out = df.copy()
    drafted = out["Pick"].lt(263)
    out["y_dr"] = np.nan
    ranks = out.loc[drafted].groupby("Year")["CarAV"].rank(pct=True)
    out.loc[drafted, "y_dr"] = ranks
    immature = out.groupby("Year")["CarAV"].transform("max").eq(0)
    out.loc[immature, "y_dr"] = np.nan
    return out


def bust_label(df: pd.DataFrame) -> pd.Series:
    return (df["y_dr"] < BUST_THRESHOLD).astype(int)


def make_pick_bust_baseline(train: pd.DataFrame):
    fit = train[train["y_dr"].notna() & train["Pick"].lt(263)]
    iso = IsotonicRegression(out_of_bounds="clip").fit(fit["Pick"], bust_label(fit))

    def predict(part: pd.DataFrame) -> np.ndarray:
        return iso.predict(part["Pick"].fillna(263))

    return predict


def fit_bust_model(train: pd.DataFrame, feats: list[str], seeds=(1, 2, 3, 4, 5)):
    fit = train[train["y_dr"].notna() & train["Pick"].le(TOP_PICK_CUTOFF)].copy()
    target = bust_label(fit)
    models = []
    for seed in seeds:
        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.03,
            num_leaves=7,
            min_data_in_leaf=60,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            n_estimators=400,
            verbose=-1,
            random_state=seed,
        )
        model.fit(fit[feats + CATS], target, categorical_feature=CATS)
        models.append(model)

    def predict(part: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(part[feats + CATS])[:, 1] for m in models], axis=0)

    return predict, models


def evaluate_test_year(df: pd.DataFrame, test_year: int, feats: list[str]) -> dict:
    train_raw = df[df["Year"] < test_year].copy()
    test_raw = df[df["Year"] == test_year].copy()
    if train_raw.empty or test_raw.empty:
        raise ValueError(f"no data for {test_year}")

    train, test = prepare_fold(train_raw, test_raw)
    used = [f for f in feats if f in train.columns]

    base = make_pick_bust_baseline(train)
    model, _ = fit_bust_model(train, used)

    scope = test[test["y_dr"].notna() & test["Pick"].le(TOP_PICK_CUTOFF)].copy()
    if len(scope) < 30:
        raise ValueError(f"too few top-{TOP_PICK_CUTOFF} rows in {test_year}")
    label = bust_label(scope)
    p_base = base(scope)
    p_model = 0.5 * np.asarray(p_base) + 0.5 * model(scope)  # anchor to slot base rate

    def flag_precision(p: np.ndarray, k: int = 10) -> float:
        idx = np.argsort(-p)[:k]
        return float(label.iloc[idx].mean())

    return {
        "test_year": test_year,
        "n_top100": int(len(scope)),
        "bust_rate": float(label.mean()),
        "auc_pick_only": float(roc_auc_score(label, p_base)) if label.nunique() > 1 else np.nan,
        "auc_model": float(roc_auc_score(label, p_model)) if label.nunique() > 1 else np.nan,
        "flag10_precision_pick_only": flag_precision(np.asarray(p_base)),
        "flag10_precision_model": flag_precision(p_model),
    }


def trait_table(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Bust rate by trait bucket across all mature top-100 picks."""
    prepared, _ = prepare_fold(df, df)
    scope = prepared[prepared["y_dr"].notna() & prepared["Pick"].le(TOP_PICK_CUTOFF)].copy()
    scope["bust"] = bust_label(scope)
    base = float(scope["bust"].mean())
    rows = []
    for feat in feats:
        if feat not in scope.columns:
            continue
        values = pd.to_numeric(scope[feat], errors="coerce")
        if values.notna().sum() < 300 or values.nunique() < 5:
            continue
        try:
            buckets = pd.qcut(values, q=5, duplicates="drop")
        except ValueError:
            continue
        grouped = scope.groupby(buckets, observed=True)["bust"].agg(["mean", "size"])
        worst = grouped["mean"].idxmax()
        best = grouped["mean"].idxmin()
        rows.append(
            {
                "trait": feat,
                "base_bust_rate": round(base, 4),
                "worst_bucket": str(worst),
                "worst_bucket_bust_rate": round(float(grouped.loc[worst, "mean"]), 4),
                "best_bucket": str(best),
                "best_bucket_bust_rate": round(float(grouped.loc[best, "mean"]), 4),
                "spread": round(float(grouped["mean"].max() - grouped["mean"].min()), 4),
                "n": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("spread", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2011)
    parser.add_argument("--last-test-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    df = add_drafted_percentile(load_dataset(end_year=args.end_year))
    feats = sorted(set(model_features_for("profile") + TRAIT_FEATURES))

    rows = []
    for year in range(args.first_test_year, args.last_test_year + 1):
        try:
            row = evaluate_test_year(df, year, feats)
        except ValueError as exc:
            print(f"Skipping {year}: {exc}")
            continue
        rows.append(row)
        print(
            f"{year}: bust_rate={row['bust_rate']:.2f} AUC pick={row['auc_pick_only']:.3f} "
            f"model={row['auc_model']:.3f} flag10 pick={row['flag10_precision_pick_only']:.2f} "
            f"model={row['flag10_precision_model']:.2f}"
        )

    summary = pd.DataFrame(rows)
    traits = trait_table(df, feats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    summary.round(4).to_csv(out_dir / "bust_model_backtest.csv", index=False)
    traits.to_csv(out_dir / "bust_trait_table.csv", index=False)

    report = {
        "definition": f"bust = within-class career percentile < {BUST_THRESHOLD} among top-{TOP_PICK_CUTOFF} picks",
        "years_evaluated": int(len(summary)),
        "mean_auc_pick_only": float(summary["auc_pick_only"].mean()) if not summary.empty else np.nan,
        "mean_auc_model": float(summary["auc_model"].mean()) if not summary.empty else np.nan,
        "mean_flag10_precision_pick_only": float(summary["flag10_precision_pick_only"].mean()) if not summary.empty else np.nan,
        "mean_flag10_precision_model": float(summary["flag10_precision_model"].mean()) if not summary.empty else np.nan,
        "auc_win_years": int((summary["auc_model"] > summary["auc_pick_only"]).sum()) if not summary.empty else 0,
        "top_traits_by_spread": traits.head(8).to_dict("records"),
    }
    (out_dir / "bust_model_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
