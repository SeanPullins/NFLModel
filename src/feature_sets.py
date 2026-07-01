"""Named feature sets for APEX model variants.

The public/default model uses `profile`: combine/profile features plus age and
college encoding. NCAA production remains available for experiments and
ablations, but is not part of the default headline model.
"""
from __future__ import annotations

from pipeline import ATHLETIC_FEATURES, COLLEGE_PRODUCTION_FEATURES, CONSENSUS_MARKET_FEATURES

OFFENSIVE_PRODUCTION_FEATURES = [
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
    "college_offensive_yds_pg",
    "college_total_td_pg",
]

DEFENSIVE_PRODUCTION_FEATURES = [
    "college_games",
    "college_tackles_pg",
    "college_sacks_pg",
    "college_ints_pg",
    "college_fumbles_pg",
    "college_def_playmaking_pg",
]

FEATURE_SETS: dict[str, list[str]] = {
    "profile": ATHLETIC_FEATURES,
    "profile_plus_production": ATHLETIC_FEATURES + COLLEGE_PRODUCTION_FEATURES,
    "production_only": COLLEGE_PRODUCTION_FEATURES,
    "offensive_production_only": OFFENSIVE_PRODUCTION_FEATURES,
    "defensive_production_only": DEFENSIVE_PRODUCTION_FEATURES,
    # Post-draft-only experiment: consensus_vs_pick compares the pre-draft
    # consensus board to the actual pick, so this set must never be used for
    # pre-draft forecasting.
    "profile_plus_consensus": ATHLETIC_FEATURES + CONSENSUS_MARKET_FEATURES,
}

DEFAULT_FEATURE_SET = "profile"
EXPERIMENTAL_PRODUCTION_FEATURE_SET = "profile_plus_production"


def raw_features_for(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set '{feature_set}'. Available: {', '.join(sorted(FEATURE_SETS))}")
    return FEATURE_SETS[feature_set]


def model_features_for(feature_set: str) -> list[str]:
    return [f"{feature}_z" for feature in raw_features_for(feature_set)] + ["age", "col_enc"]
