"""Train APEX and build the public board.

Default public model:
    raw APEX with the `profile` feature set

That means combine/profile features + age + college encoding. NCAA production
features remain available for ablations/experiments, but are not included in the
headline board unless explicitly promoted later.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS, model_features_for
from pipeline import (
    CATS,
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

HOLDOUT_TRAIN_END = 2009
HOLDOUT_VALID_START = 2010
HOLDOUT_VALID_END = 2011
HOLDOUT_TEST_START = 2012
HOLDOUT_TEST_END = 2014
DEFAULT_BOARD_END_YEAR = 2026


def evaluate_holdout(df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> tuple[pd.DataFrame, dict, dict]:
    """Run the original 2012-2014 benchmark with fold-safe transforms."""
    feats = model_features_for(feature_set)
    train_raw = df[df["Year"] <= HOLDOUT_TRAIN_END].copy()
    valid_raw = df[(df["Year"] >= HOLDOUT_VALID_START) & (df["Year"] <= HOLDOUT_VALID_END)].copy()
    test_raw = df[(df["Year"] >= HOLDOUT_TEST_START) & (df["Year"] <= HOLDOUT_TEST_END)].copy()

    train, valid = prepare_fold(train_raw, valid_raw)
    base, _, _ = make_baseline(train)
    resid, _ = make_resid(train, base, feats=feats)
    shrink = tune_position_shrinkage(valid, base, resid)

    # Final fit uses all pre-test data after shrinkage was selected on 2010-2011.
    train2_raw = df[df["Year"] <= HOLDOUT_VALID_END].copy()
    train2, test, feature_stats = prepare_fold(train2_raw, test_raw, return_stats=True)
    pick_only, _ = make_pick_baseline(train2)
    base2, _, _ = make_baseline(train2)
    resid2, _ = make_resid(train2, base2, feats=feats)

    scored = test.copy()
    scored["pick_only"] = pick_only(scored)
    scored["pos_base"] = base2(scored)
    scored["apex"] = score_apex(scored, base2, resid2, shrink)

    metrics = {
        "feature_set": feature_set,
        "public_model": "raw_apex_profile_only",
        "window": {
            "train": f"<= {HOLDOUT_VALID_END}",
            "validation_for_shrinkage": f"{HOLDOUT_VALID_START}-{HOLDOUT_VALID_END}",
            "test": f"{HOLDOUT_TEST_START}-{HOLDOUT_TEST_END}",
        },
        "pick_only": metric_row(scored, "pick_only"),
        "pos_base": metric_row(scored, "pos_base"),
        "apex": metric_row(scored, "apex"),
        "position": {},
        "shrink": shrink,
    }

    drafted = scored[scored["Pick"].lt(263)].copy()
    for pos, group in drafted.groupby("pos_g", observed=True):
        if len(group) >= 60:
            metrics["position"][str(pos)] = {
                "n": int(len(group)),
                "apex_spearman": safe_spearman(group["apex"], group["y"]),
                "pos_base_spearman": safe_spearman(group["pos_base"], group["y"]),
                "pick_only_spearman": safe_spearman(group["pick_only"], group["y"]),
                "shrink": shrink.get(str(pos), 0.4),
            }

    return scored, metrics, {"feature_stats": feature_stats, "shrink": shrink, "features": feats}


def fit_production_board(df: pd.DataFrame, shrink: dict[str, float], feature_set: str = DEFAULT_FEATURE_SET) -> tuple[pd.DataFrame, dict]:
    """Fit on all available historical data and write the model board."""
    feats = model_features_for(feature_set)
    train, scored, feature_stats = prepare_fold(df, df, return_stats=True)
    base, glob, isos = make_baseline(train)
    resid, models = make_resid(train, base, feats=feats)

    scored = scored.copy()
    scored["apex"] = score_apex(scored, base, resid, shrink)
    scored["exp_at_pick"] = base(scored)
    scored["talent_resid"] = resid(scored)
    scored["surplus"] = scored["apex"] - scored["exp_at_pick"]

    out = pd.DataFrame(
        {
            "Year": scored["Year"],
            "Player": scored["Player"],
            "Pos": scored["Pos"],
            "pos_g": scored["pos_g"].astype(str),
            "College": scored["college"],
            "Pick": scored["Pick"],
            "Rnd": scored.get("Rnd", np.nan),
            "CarAV": scored["CarAV"],
            "y": scored["y"],
            "apex": scored["apex"],
            "exp_at_pick": scored["exp_at_pick"],
            "talent_resid": scored["talent_resid"],
            "surplus": scored["surplus"],
            "feature_set": feature_set,
            "model_status": "headline_profile_only_raw_apex",
        }
    )

    artifacts = {
        "glob": glob,
        "isos": isos,
        "shrink": shrink,
        "feature_stats": feature_stats,
        "resid_models": models,
        "features": feats,
        "feature_set": feature_set,
    }
    return out, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", type=str, default=DEFAULT_FEATURE_SET, choices=sorted(FEATURE_SETS))
    parser.add_argument("--end-year", type=int, default=DEFAULT_BOARD_END_YEAR)
    args = parser.parse_args()

    df = load_dataset(end_year=args.end_year)
    holdout_scored, metrics, fit_objects = evaluate_holdout(df, args.feature_set)

    reports_dir = ROOT / "reports"
    data_dir = ROOT / "data"
    models_dir = ROOT / "models"
    reports_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    (reports_dir / "holdout_2012_2014_metrics.json").write_text(json.dumps(metrics, indent=2))
    holdout_scored.round(4).to_csv(reports_dir / "holdout_2012_2014_scored.csv", index=False)

    board, artifacts = fit_production_board(df, fit_objects["shrink"], args.feature_set)
    board.round(4).to_csv(data_dir / "apex_board.csv", index=False)

    for i, model in enumerate(artifacts["resid_models"]):
        model.booster_.save_model(str(models_dir / f"apex_resid_{i}.txt"))

    joblib.dump(
        {
            "glob": artifacts["glob"],
            "isos": artifacts["isos"],
            "shrink": artifacts["shrink"],
            "feature_stats": artifacts["feature_stats"],
            "features": artifacts["features"],
            "feature_set": artifacts["feature_set"],
            "categoricals": CATS,
        },
        models_dir / "apex_baseline_and_transforms.pkl",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {data_dir / 'apex_board.csv'} using feature_set={args.feature_set}")
    print(f"Wrote reports to {reports_dir}")
    print(f"Wrote models to {models_dir}")


if __name__ == "__main__":
    main()
