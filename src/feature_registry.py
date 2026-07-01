"""Feature definitions for future APEX upgrades.

The registry keeps optional production and consensus inputs explicit so new data
can be added without silently changing the production model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline import ROOT

KEY_COLUMNS = ["Year", "Player"]
OPTIONAL_ID_COLUMNS = ["Pos", "pos_g", "College", "college", "Pick", "Rnd"]

PRODUCTION_DIR = ROOT / "data" / "production"
CONSENSUS_DIR = ROOT / "data" / "consensus"
ENRICHED_FEATURE_FILE = ROOT / "data" / "model_features.csv"
FEATURE_COVERAGE_REPORT = ROOT / "reports" / "feature_coverage.json"


@dataclass(frozen=True)
class FeatureFileSpec:
    name: str
    path: Path
    feature_columns: tuple[str, ...]
    description: str


PRODUCTION_FEATURES_BY_GROUP: dict[str, tuple[str, ...]] = {
    "QB": (
        "qb_starts",
        "qb_age_adj_epa_per_play",
        "qb_cpoe",
        "qb_adj_completion_pct",
        "qb_pressure_to_sack_rate",
        "qb_sack_rate",
        "qb_big_time_throw_rate",
        "qb_turnover_worthy_play_rate",
        "qb_rush_epa_per_game",
        # PFF passing features (src/build_pff_features.py; local only, licensed)
        "qb_pff_pass_grade",
        "qb_pff_pass_grade_final",
        "qb_pff_dropbacks",
        # ESPN college Total QBR features (src/build_qb_production.py)
        "qb_career_plays",
        "qb_seasons",
        "qb_final_qbr",
        "qb_best_qbr",
        "qb_epa_per_play",
        "qb_final_epa_per_play",
        "qb_sack_epa_per_play",
        "qb_run_epa_per_play",
    ),
    "WR": (
        "wr_yards_per_route_run",
        "wr_target_share",
        "wr_dominator",
        "wr_breakout_age",
        "wr_explosive_reception_rate",
        "wr_contested_catch_rate",
        "wr_slot_rate",
    ),
    "RB": (
        "rb_yards_after_contact_per_attempt",
        "rb_missed_tackles_forced_rate",
        "rb_explosive_run_rate",
        "rb_receiving_share",
        "rb_yards_per_route_run",
    ),
    "TE": (
        "te_yards_per_route_run",
        "te_target_share",
        "te_receiving_share",
        "te_inline_rate",
        "te_slot_rate",
    ),
    "OL": (
        "ol_pressure_rate_allowed",
        "ol_blown_block_rate",
        "ol_pass_block_grade",
        "ol_run_block_grade",
        "ol_snaps",
    ),
    "EDGE": (
        "edge_pressure_rate",
        "edge_pass_rush_win_rate",
        "edge_sack_to_pressure_rate",
        "edge_run_stop_rate",
    ),
    "DT": (
        "dt_pressure_rate",
        "dt_pass_rush_win_rate",
        "dt_run_stop_rate",
        "dt_sack_to_pressure_rate",
    ),
    "LB": (
        "lb_run_stop_rate",
        "lb_missed_tackle_rate",
        "lb_coverage_grade",
        "lb_pressure_rate",
    ),
    "DB": (
        "db_yards_per_coverage_snap",
        "db_forced_incompletion_rate",
        "db_interception_rate",
        "db_missed_tackle_rate",
        "db_coverage_grade",
    ),
}

CONSENSUS_FEATURES: tuple[str, ...] = (
    "consensus_rank",
    "espn_grade",
    "espn_pos_rank",
    "expected_pick",
    "mock_avg_pick",
    "mock_pick_std",
    "n_mocks",
    "n_big_boards",
    "nfl_com_grade",
    "recruiting_stars",
    "recruiting_rank",
    "combine_invite",
    "senior_bowl",
    "shrine_bowl",
)

FEATURE_FILE_SPECS: tuple[FeatureFileSpec, ...] = (
    FeatureFileSpec(
        name="qb_production",
        path=PRODUCTION_DIR / "qb_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["QB"],
        description="Quarterback production and efficiency metrics.",
    ),
    FeatureFileSpec(
        name="wr_production",
        path=PRODUCTION_DIR / "wr_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["WR"],
        description="Wide receiver production, target, and route metrics.",
    ),
    FeatureFileSpec(
        name="rb_production",
        path=PRODUCTION_DIR / "rb_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["RB"],
        description="Running back rushing and receiving production metrics.",
    ),
    FeatureFileSpec(
        name="te_production",
        path=PRODUCTION_DIR / "te_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["TE"],
        description="Tight end receiving and alignment metrics.",
    ),
    FeatureFileSpec(
        name="ol_production",
        path=PRODUCTION_DIR / "ol_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["OL"],
        description="Offensive line pressure, blocking, and snap metrics.",
    ),
    FeatureFileSpec(
        name="edge_production",
        path=PRODUCTION_DIR / "edge_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["EDGE"],
        description="EDGE pass-rush and run-defense metrics.",
    ),
    FeatureFileSpec(
        name="dt_production",
        path=PRODUCTION_DIR / "dt_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["DT"],
        description="Interior defensive line production metrics.",
    ),
    FeatureFileSpec(
        name="lb_production",
        path=PRODUCTION_DIR / "lb_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["LB"],
        description="Linebacker run, coverage, pressure, and tackle metrics.",
    ),
    FeatureFileSpec(
        name="db_production",
        path=PRODUCTION_DIR / "db_production.csv",
        feature_columns=PRODUCTION_FEATURES_BY_GROUP["DB"],
        description="Defensive back coverage and tackling metrics.",
    ),
    FeatureFileSpec(
        name="consensus_board",
        path=CONSENSUS_DIR / "consensus_board.csv",
        feature_columns=CONSENSUS_FEATURES,
        description="Pre-draft market features: consensus ranks, mock draft expected pick, and event flags.",
    ),
)

POSITION_MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "SKILL": ("WR", "RB", "TE"),
    "OL": ("OL",),
    "FRONT": ("EDGE", "DT"),
    "LB": ("LB",),
    "DB": ("DB",),
}


def production_features_for_pos(pos_g: str) -> list[str]:
    return list(PRODUCTION_FEATURES_BY_GROUP.get(str(pos_g), ()))


def all_optional_features() -> list[str]:
    cols: list[str] = []
    for spec in FEATURE_FILE_SPECS:
        cols.extend(spec.feature_columns)
    return sorted(set(cols))


def consensus_market_features() -> list[str]:
    return list(CONSENSUS_FEATURES)
