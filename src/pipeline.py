"""APEX Draft Model utilities.

This version keeps every learned transform inside the active training fold:
college encodings, athletic z-scores, production z-scores, and market baselines
are fit on past data only before scoring validation/test rows.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import joblib  # kept for downstream artifact compatibility
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
LEGACY_DATA_DIR = Path("/home/claude/kv/NFL-Player-Valuation-master/data")

POSMAP = {
    "HB": "RB", "FB": "RB", "TB": "RB", "RB": "RB",
    "S": "DB", "FS": "DB", "SS": "DB", "SAF": "DB", "CB": "DB", "DB": "DB",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "DE": "EDGE", "EDGE": "EDGE", "NT": "DT", "DT": "DT",
    "C": "OL", "G": "OL", "OG": "OL", "OT": "OL", "T": "OL", "OL": "OL",
    "QB": "QB", "WR": "WR", "TE": "TE",
    "K": "ST", "P": "ST", "LS": "ST", "PK": "ST", "PN": "ST",
}

ATHLETIC_FEATURES = [
    "dash", "speed_score", "explosion", "agility", "weight",
    "height", "bmi", "bench", "vert_leap", "broad",
]

# Pre-draft college production features from phcs971/nfl-draft-dataset.
# These intentionally exclude NFL career fields to avoid outcome leakage.
COLLEGE_PRODUCTION_FEATURES = [
    "college_games",
    "college_pass_yds_pg",
    "college_pass_td_pg",
    "college_pass_int_pg",
    "college_pass_cmp_pct",
    "college_pass_td_int_ratio",
    "college_rush_yds_pg",
    "college_rush_td_pg",
    "college_rec_yds_pg",
    "college_rec_td_pg",
    "college_tackles_pg",
    "college_sacks_pg",
    "college_ints_pg",
    "college_fumbles_pg",
    "college_offensive_yds_pg",
    "college_total_td_pg",
    "college_def_playmaking_pg",
]

# Pre-draft consensus market features from data/consensus/consensus_board.csv
# (ESPN pre-draft boards via JackLich10/nfl-draft-data). log_consensus_rank and
# espn_grade are pure pre-draft signals; consensus_vs_pick compares the
# pre-draft board to the actual pick, so it is post-draft-only.
CONSENSUS_MARKET_FEATURES = [
    "log_consensus_rank",
    "espn_grade",
    "consensus_vs_pick",
]

RAW_FEATURES = ATHLETIC_FEATURES + COLLEGE_PRODUCTION_FEATURES + CONSENSUS_MARKET_FEATURES
# FEATS_A intentionally excludes consensus features: it is the default residual
# feature list for the pre-draft model, where consensus_vs_pick would leak the
# actual pick.
FEATS_A = [f"{c}_z" for c in ATHLETIC_FEATURES + COLLEGE_PRODUCTION_FEATURES] + ["age", "col_enc"]
FEATS_C = FEATS_A + ["logpick"]
CATS = ["pos_g"]

OFF_BOARD_CONSENSUS_RANK = 450.0


def norm(value: object) -> str:
    text = str(value).lower().replace("jr", "").replace("iii", "").replace("ii", "")
    return re.sub(r"[^a-z]", "", text)


def resolve_data_file(filename: str, data_dir: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir) / filename)
    if os.environ.get("APEX_DATA_DIR"):
        candidates.append(Path(os.environ["APEX_DATA_DIR"]) / filename)
    candidates += [
        DEFAULT_DATA_DIR / filename,
        Path(__file__).resolve().parent / filename,
        LEGACY_DATA_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n  - {searched}")


def _ensure_columns(df: pd.DataFrame, columns: list[str], default=np.nan) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = default
    return df


def _coerce_combine_columns(cb: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "Year" in cb.columns and "year" not in cb.columns:
        rename["Year"] = "year"
    if "Player" in cb.columns and "player" not in cb.columns:
        rename["Player"] = "player"
    if "Pos" in cb.columns and "position" not in cb.columns:
        rename["Pos"] = "position"
    if "College" in cb.columns and "college" not in cb.columns:
        rename["College"] = "college"
    return cb.rename(columns=rename)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = [
        "height", "weight", "dash", "vert_leap", "bench", "broad",
        "cone", "shuttle", "Pick", "CarAV", *COLLEGE_PRODUCTION_FEATURES,
    ]
    _ensure_columns(out, numeric)
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if "Age" in out.columns and "age" not in out.columns:
        out["age"] = pd.to_numeric(out["Age"], errors="coerce")
    elif "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    else:
        out["age"] = np.nan

    out["bmi"] = out["weight"] / (out["height"] ** 2) * 703
    out["speed_score"] = np.where(
        out["dash"].gt(0), (out["weight"] * 200) / (out["dash"] ** 4), np.nan
    )
    out["explosion"] = out["vert_leap"] + out["broad"] / 12
    out["agility"] = out["cone"] + out["shuttle"]
    out["logpick"] = np.log(out["Pick"].clip(1, 262))

    consensus_rank = pd.to_numeric(out.get("consensus_rank", np.nan), errors="coerce")
    out["log_consensus_rank"] = np.log(consensus_rank.clip(1, OFF_BOARD_CONSENSUS_RANK))
    out["espn_grade"] = pd.to_numeric(out.get("espn_grade", np.nan), errors="coerce")
    out["consensus_vs_pick"] = out["logpick"] - out["log_consensus_rank"]
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CarAV"] = pd.to_numeric(out["CarAV"], errors="coerce").fillna(0.0)
    out["y"] = out.groupby("Year")["CarAV"].rank(pct=True)
    out["hit"] = (
        out.groupby("Year")["CarAV"].rank(pct=True, ascending=False) <= 0.10
    ).astype(int)
    # Classes with no recorded career value yet (e.g. just-drafted years) have
    # no real outcome. Ranking all-zero CarAV would fabricate identical
    # mid-pack "actual" percentiles, so mask the target for those classes.
    immature = out.groupby("Year")["CarAV"].transform("max").eq(0)
    out.loc[immature, "y"] = np.nan
    out.loc[immature, "hit"] = 0
    return out


def load_dataset(
    data_dir: str | Path | None = None,
    start_year: int = 2000,
    end_year: int = 2016,
    include_undrafted: bool = True,
) -> pd.DataFrame:
    dr = pd.read_csv(resolve_data_file("draft_data.csv", data_dir))
    cb = _coerce_combine_columns(pd.read_csv(resolve_data_file("combine_data_pfr_with_stats.csv", data_dir)))

    dr = dr[(dr["Year"] >= start_year) & (dr["Year"] <= end_year)].copy()
    cb = cb[(cb["year"] >= start_year) & (cb["year"] <= end_year)].copy()

    dr["key"] = dr["Player"].map(norm) + "_" + dr["Year"].astype(str)
    dr["pos_g"] = dr["Pos"].map(POSMAP).fillna("OTH")
    dr["CarAV"] = pd.to_numeric(dr.get("CarAV", 0), errors="coerce").fillna(0)

    merge_metrics = [
        "height", "weight", "dash", "vert_leap", "bench", "broad", "cone", "shuttle",
        *COLLEGE_PRODUCTION_FEATURES,
    ]
    _ensure_columns(cb, merge_metrics)
    for column in merge_metrics:
        cb[column] = pd.to_numeric(cb[column], errors="coerce").replace(0, np.nan if column in ATHLETIC_FEATURES else 0)
    cb["key"] = cb["player"].map(norm) + "_" + cb["year"].astype(str)
    cb = cb.drop_duplicates("key")

    merge_cols = ["key", *merge_metrics] + (["college"] if "college" in cb.columns else [])
    df = dr.merge(cb[merge_cols], on="key", how="left")

    if include_undrafted:
        ud = cb[~cb["key"].isin(dr["key"])].copy()
        ud = ud.rename(columns={"year": "Year", "player": "Player", "position": "Pos"})
        ud["pos_g"] = ud["Pos"].map(POSMAP).fillna("OTH")
        ud["CarAV"] = 0.0
        ud["Pick"] = 263.0
        ud["Rnd"] = 8
        ud = drop_phantom_undrafted(ud, dr)
        keep = ["Year", "Player", "Pos", "pos_g", "Pick", "Rnd", "CarAV", "key", *merge_metrics]
        if "college" in ud.columns:
            keep.append("college")
        df = pd.concat([df, ud[keep]], ignore_index=True)

    if "college" not in df.columns:
        df["college"] = "Unknown"
    df["college"] = df["college"].fillna("Unknown")
    df = df[df["pos_g"] != "ST"].copy()
    df = merge_consensus_board(df, data_dir=data_dir)
    df = add_targets(add_base_features(df))
    df["pos_g"] = df["pos_g"].astype("category")
    return df


def merge_consensus_board(df: pd.DataFrame, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Attach pre-draft consensus board columns when the optional file exists.

    Players missing from the board in a year the board covers are treated as
    "off board" (rank OFF_BOARD_CONSENSUS_RANK). Years the board does not cover
    are left as NaN so training-fold median fill handles them.
    """
    out = df.copy()
    try:
        path = resolve_data_file("consensus/consensus_board.csv", data_dir)
    except FileNotFoundError:
        out["consensus_rank"] = np.nan
        out["espn_grade"] = np.nan
        return out

    board = pd.read_csv(path)
    board = board[board["Year"].notna() & board["Player"].notna()].copy()
    board["key"] = board["Player"].map(norm) + "_" + pd.to_numeric(board["Year"], errors="coerce").astype("Int64").astype(str)
    keep = board.drop_duplicates("key")[["key", "consensus_rank", "espn_grade"]]
    out = out.merge(keep, on="key", how="left", suffixes=("", "_board"))

    board_years = set(pd.to_numeric(board["Year"], errors="coerce").dropna().astype(int))
    covered = out["Year"].isin(board_years)
    out.loc[covered & out["consensus_rank"].isna(), "consensus_rank"] = OFF_BOARD_CONSENSUS_RANK
    return out


def drop_phantom_undrafted(ud: pd.DataFrame, dr: pd.DataFrame) -> pd.DataFrame:
    """Remove stale combine/pro-day rows that duplicate a drafted player.

    Measurement sources sometimes list a player under a neighboring year (e.g.
    a 2024 draftee also appearing as a 2025 "prospect"). If an undrafted row
    shares a normalized name and position group with a drafted player within
    two years, treat it as a phantom duplicate rather than a new prospect.
    """
    drafted = dr[["Player", "Year", "pos_g"]].copy()
    drafted["name_key"] = drafted["Player"].map(norm)
    drafted_index = set(
        (row.name_key, int(row.Year) + offset, str(row.pos_g))
        for row in drafted.itertuples()
        for offset in (-2, -1, 0, 1, 2)
    )
    name_keys = ud["Player"].map(norm)
    mask_phantom = [
        (name_key, int(year), str(pos_g)) in drafted_index
        for name_key, year, pos_g in zip(name_keys, ud["Year"], ud["pos_g"])
    ]
    n_dropped = int(sum(mask_phantom))
    if n_dropped:
        print(f"Dropped {n_dropped} phantom undrafted rows duplicating drafted players in neighboring years")
    return ud.loc[[not m for m in mask_phantom]].copy()


def college_enc(train: pd.DataFrame, part: pd.DataFrame, k: int = 12) -> pd.Series:
    stats = train.groupby(train["college"].fillna("Unknown"))["y"].agg(["mean", "count"])
    prior = float(train["y"].mean())
    enc = (stats["mean"] * stats["count"] + prior * k) / (stats["count"] + k)
    return part["college"].fillna("Unknown").map(enc).fillna(prior)


def fit_feature_stats(train: pd.DataFrame, features: list[str] | None = None) -> dict:
    features = features or RAW_FEATURES
    pos = train["pos_g"].astype(str)
    stats = {"global_median": {}, "global_mu": {}, "global_sd": {}, "pos_mu": {}, "pos_sd": {}}

    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        filled = values.fillna(median)
        gmu = float(filled.mean())
        gsd = float(filled.std(ddof=0))
        if not np.isfinite(gsd) or gsd == 0:
            gsd = 1.0

        tmp = pd.DataFrame({"pos": pos, "value": filled})
        pmu = tmp.groupby("pos", observed=True)["value"].mean().to_dict()
        psd = tmp.groupby("pos", observed=True)["value"].std(ddof=0).to_dict()
        psd = {p: (1.0 if not np.isfinite(sd) or sd == 0 else float(sd)) for p, sd in psd.items()}

        stats["global_median"][feature] = median
        stats["global_mu"][feature] = gmu
        stats["global_sd"][feature] = gsd
        stats["pos_mu"][feature] = {p: float(v) for p, v in pmu.items()}
        stats["pos_sd"][feature] = psd
    return stats


def apply_feature_stats(part: pd.DataFrame, stats: dict, features: list[str] | None = None) -> pd.DataFrame:
    features = features or RAW_FEATURES
    out = part.copy()
    pos = out["pos_g"].astype(str)
    for feature in features:
        median = stats["global_median"].get(feature, 0.0)
        values = pd.to_numeric(out.get(feature, median), errors="coerce").fillna(median)
        z = pd.Series(index=out.index, dtype=float)
        for p in pos.unique():
            mask = pos.eq(p)
            mu = stats["pos_mu"].get(feature, {}).get(p, stats["global_mu"].get(feature, 0.0))
            sd = stats["pos_sd"].get(feature, {}).get(p, stats["global_sd"].get(feature, 1.0))
            if not np.isfinite(sd) or sd == 0:
                sd = 1.0
            z.loc[mask] = (values.loc[mask] - mu) / sd
        out[f"{feature}_z"] = z.clip(-4, 4)
    return out


def align_pos_categories(train: pd.DataFrame, part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align category dtype without failing when OTH already exists.

    Unknown/previously unseen position groups are mapped to OTH. Older code
    always added OTH after casting, which raises a ValueError when OTH is already
    present in the training categories. Because the backtest loop catches
    ValueError, those years were silently skipped.
    """
    categories = sorted(set(train["pos_g"].astype(str).dropna()).union({"OTH"}))
    dtype = CategoricalDtype(categories=categories)
    out_train = train.copy()
    out_part = part.copy()
    out_train["pos_g"] = out_train["pos_g"].astype(str).where(lambda s: s.isin(categories), "OTH").astype(dtype)
    out_part["pos_g"] = out_part["pos_g"].astype(str).where(lambda s: s.isin(categories), "OTH").astype(dtype)
    return out_train, out_part


def prepare_fold(
    train_raw: pd.DataFrame,
    part_raw: pd.DataFrame,
    *,
    return_stats: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = train_raw.copy()
    part = part_raw.copy()
    train, part = align_pos_categories(train, part)
    train["col_enc"] = college_enc(train, train)
    part["col_enc"] = college_enc(train, part)
    stats = fit_feature_stats(train)
    train = apply_feature_stats(train, stats)
    part = apply_feature_stats(part, stats)
    if return_stats:
        return train, part, stats
    return train, part


def make_pick_baseline(train: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], np.ndarray], IsotonicRegression]:
    fit = train[train["Pick"].notna() & train["y"].notna()].copy()
    iso = IsotonicRegression(out_of_bounds="clip").fit(-fit["Pick"], fit["y"])

    def predict(part: pd.DataFrame) -> np.ndarray:
        return iso.predict(-part["Pick"].fillna(263))

    return predict, iso


def make_baseline(
    train: pd.DataFrame,
    *,
    min_pos_samples: int = 300,
    pos_blend: float = 0.5,
) -> tuple[Callable[[pd.DataFrame], np.ndarray], IsotonicRegression, dict[str, IsotonicRegression]]:
    glob_predict, glob = make_pick_baseline(train)
    isos = {}
    fit = train[train["Pick"].notna() & train["y"].notna()]
    for pos, group in fit.groupby("pos_g", observed=True):
        if len(group) >= min_pos_samples:
            isos[str(pos)] = IsotonicRegression(out_of_bounds="clip").fit(-group["Pick"], group["y"])

    def predict(part: pd.DataFrame) -> np.ndarray:
        out = glob_predict(part)
        for pos, iso in isos.items():
            mask = part["pos_g"].astype(str).eq(pos) & part["Pick"].notna()
            if mask.any():
                pos_pred = iso.predict(-part.loc[mask, "Pick"])
                out[mask.to_numpy()] = (1 - pos_blend) * out[mask.to_numpy()] + pos_blend * pos_pred
        return out

    return predict, glob, isos


def make_resid(
    train: pd.DataFrame,
    base: Callable[[pd.DataFrame], np.ndarray],
    *,
    feats: list[str] | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> tuple[Callable[[pd.DataFrame], np.ndarray], list[lgb.LGBMRegressor]]:
    feats = feats or FEATS_A
    fit = train[train["y"].notna()].copy()
    target = fit["y"].to_numpy() - base(fit)
    models = []
    for seed in seeds:
        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.02,
            num_leaves=15,
            min_data_in_leaf=80,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            n_estimators=600,
            verbose=-1,
            random_state=seed,
        )
        model.fit(fit[feats + CATS], target, categorical_feature=CATS)
        models.append(model)

    def predict(part: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(part[feats + CATS]) for m in models], axis=0)

    return predict, models


def safe_spearman(pred, actual) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["pred"].nunique() < 2 or frame["actual"].nunique() < 2:
        return np.nan
    return float(spearmanr(frame["pred"], frame["actual"]).statistic)


def safe_auc(actual_hit, pred) -> float:
    frame = pd.DataFrame({"pred": pred, "hit": actual_hit}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["hit"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(frame["hit"], frame["pred"]))


def precision_at_k(df: pd.DataFrame, score_col: str, k: int) -> float:
    usable = df[[score_col, "hit"]].replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        return np.nan
    return float(usable.nlargest(min(k, len(usable)), score_col)["hit"].mean())


def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int) -> float:
    usable = df[[score_col, "y"]].replace([np.inf, -np.inf], np.nan).dropna()
    if usable.empty:
        return np.nan
    k = min(k, len(usable))
    ranked = usable.nlargest(k, score_col)["y"].to_numpy()
    ideal = usable.nlargest(k, "y")["y"].to_numpy()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    idcg = float(np.sum(ideal * discounts))
    return np.nan if idcg == 0 else float(np.sum(ranked * discounts)) / idcg


def tune_position_shrinkage(
    validation: pd.DataFrame,
    base: Callable[[pd.DataFrame], np.ndarray],
    resid: Callable[[pd.DataFrame], np.ndarray],
    *,
    candidates: np.ndarray | None = None,
    min_samples: int = 40,
    default: float = 0.4,
) -> dict[str, float]:
    candidates = candidates if candidates is not None else np.arange(0, 1.05, 0.1)
    drafted = validation[validation["Pick"].lt(263) & validation["y"].notna()].copy()
    b = base(drafted)
    r = resid(drafted)
    shrink = {}
    for pos, group in drafted.groupby("pos_g", observed=True):
        mask = drafted["pos_g"].astype(str).eq(str(pos)).to_numpy()
        if len(group) < min_samples:
            shrink[str(pos)] = default
            continue
        best_s, best_metric = default, -np.inf
        for s in candidates:
            metric = safe_spearman(b[mask] + s * r[mask], drafted.loc[mask, "y"])
            if np.isfinite(metric) and metric > best_metric:
                best_s, best_metric = float(s), metric
        shrink[str(pos)] = best_s
    return shrink


def score_apex(
    part: pd.DataFrame,
    base: Callable[[pd.DataFrame], np.ndarray],
    resid: Callable[[pd.DataFrame], np.ndarray],
    shrink: dict[str, float],
    *,
    default: float = 0.4,
) -> np.ndarray:
    b = base(part)
    r = resid(part)
    s = part["pos_g"].astype(str).map(shrink).fillna(default).to_numpy(dtype=float)
    return b + s * r


def metric_row(df: pd.DataFrame, score_col: str, prefix: str = "") -> dict[str, float]:
    drafted = df[df["Pick"].lt(263)].copy()
    return {
        f"{prefix}spearman_all": safe_spearman(df[score_col], df["y"]),
        f"{prefix}spearman_drafted": safe_spearman(drafted[score_col], drafted["y"]),
        f"{prefix}hit_auc_drafted": safe_auc(drafted["hit"], drafted[score_col]),
        f"{prefix}precision_at_32": precision_at_k(drafted, score_col, 32),
        f"{prefix}precision_at_64": precision_at_k(drafted, score_col, 64),
        f"{prefix}ndcg_at_32": ndcg_at_k(drafted, score_col, 32),
        f"{prefix}ndcg_at_64": ndcg_at_k(drafted, score_col, 64),
    }


if __name__ == "__main__":
    df = load_dataset()
    train, test = prepare_fold(df[df["Year"] <= 2011], df[(df["Year"] >= 2012) & (df["Year"] <= 2014)])
    pick, _ = make_pick_baseline(train)
    base, _, _ = make_baseline(train)
    resid, _ = make_resid(train, base)
    shrink = {str(pos): 0.4 for pos in train["pos_g"].cat.categories}
    scored = test.copy()
    scored["pick_only"] = pick(scored)
    scored["pos_base"] = base(scored)
    scored["apex"] = score_apex(scored, base, resid, shrink)
    print("Pick-only:", metric_row(scored, "pick_only"))
    print("Position baseline:", metric_row(scored, "pos_base"))
    print("APEX default shrink:", metric_row(scored, "apex"))
