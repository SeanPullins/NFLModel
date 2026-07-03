"""Write a no-spend signal scouting backlog for APEX.

This does not promote any feature into the model. It creates a ranked queue of
free/public signal families to build and validate after the current reports land.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import ROOT

OUT_JSON = ROOT / "reports" / "free_signal_backlog.json"
OUT_CSV = ROOT / "reports" / "free_signal_backlog.csv"

SIGNALS = [
    {
        "agent": "Market Disagreement Agent",
        "signal_family": "consensus_drift_and_dispersion",
        "priority": 1,
        "free_source": "public consensus/mock/board exports already supported by repo plus manual CSV drops",
        "candidate_features": "consensus_rank, expected_pick, mock_pick_std, rank drift by date, rank-biased distance, player vs pick delta",
        "why_it_might_help": "Draft slot is strongest baseline; the edge is in finding when market disagrees with actual pick or public consensus is unstable.",
        "build_next": "Extend consensus_board schema with board_date, source_count, mock_pick_std, early_vs_late_delta, and consensus_vs_pick guardrails.",
        "validation_gate": "Residual lift after log pick; improve worst-year drawdown before public score promotion.",
        "spend": 0,
    },
    {
        "agent": "CFBD PPA Agent",
        "signal_family": "player_ppa_efficiency",
        "priority": 2,
        "free_source": "CollegeFootballData /ppa/players/season and /ppa/players/games using existing repo key",
        "candidate_features": "overall_ppa, passing_ppa, rushing_ppa, receiving_ppa, final_year_ppa, career_weighted_ppa",
        "why_it_might_help": "PPA should capture efficiency/context better than raw yards and TDs.",
        "build_next": "Add build_cfbd_ppa.py and join by normalized player, team, pre-draft seasons.",
        "validation_gate": "Beat raw CFBD production in residual Spearman and hold direction across eras.",
        "spend": 0,
    },
    {
        "agent": "CFBD Usage Agent",
        "signal_family": "usage_share_and_role",
        "priority": 3,
        "free_source": "CollegeFootballData /player/usage and /player/season/overview",
        "candidate_features": "usage_overall, pass_usage, rush_usage, target_usage, final_year_usage, role_stability",
        "why_it_might_help": "Role-adjusted production can separate true focal points from stat-sheet passengers.",
        "build_next": "Add usage features by season, then career/final/peak aggregates.",
        "validation_gate": "Must add value within WR/RB/TE/QB groups after controlling for pick.",
        "spend": 0,
    },
    {
        "agent": "Recruiting Prior Agent",
        "signal_family": "recruiting_pedigree_and_talent_context",
        "priority": 4,
        "free_source": "CollegeFootballData recruiting/player and team talent endpoints where accessible",
        "candidate_features": "stars, recruit_rating, recruit_rank, team_talent, over_under_recruit_profile, late_bloomer_flag",
        "why_it_might_help": "Prior prospect pedigree can flag early talent and late bloomers, especially before full college production stabilizes.",
        "build_next": "Join recruit records by player, school, hometown, and class year; create late_bloomer and pedigree-confirmed flags.",
        "validation_gate": "Use as a shrinkage prior only; do not let high-school priors overpower college/NFL draft evidence.",
        "spend": 0,
    },
    {
        "agent": "SackSEER-Style EDGE Agent",
        "signal_family": "edge_explosive_pressure_proxy",
        "priority": 5,
        "free_source": "CFBD defensive stats plus combine/pro-day measurements from nflverse/PFR-derived releases",
        "candidate_features": "sack_rate, tfl_rate, vertical, shuttle, missed_games_proxy, sack_to_tfl, explosive_edge_flag",
        "why_it_might_help": "Historical edge models emphasize adjusted sack productivity plus explosiveness and agility, not sacks alone.",
        "build_next": "Create EDGE-only feature set with sack/TFL per game, vertical, shuttle, and availability proxies.",
        "validation_gate": "EDGE-only rolling test must beat draft slot and reduce bust precision misses.",
        "spend": 0,
    },
    {
        "agent": "Schedule Context Agent",
        "signal_family": "opponent_adjusted_production",
        "priority": 6,
        "free_source": "CFBD SP+/SRS/FPI/team ratings and opponent/team schedule endpoints",
        "candidate_features": "opponent_def_sp, opponent_off_sp, conference_strength, production_vs_top_opponents, garbage_time_filtered_flag",
        "why_it_might_help": "College stat translation depends on opponent quality, scheme, and conference context.",
        "build_next": "Aggregate player production against opponent strength buckets by season.",
        "validation_gate": "Must outperform raw production for G5/FCS/small-school translation cases.",
        "spend": 0,
    },
    {
        "agent": "Experience Curve Agent",
        "signal_family": "age_starts_and_declared_status",
        "priority": 7,
        "free_source": "draft data, roster/history pages, CFBD season participation, existing age/combine fields",
        "candidate_features": "draft_age, seasons_played, early_declare, final_year_age, qb_experience_bucket, one_year_wonder_flag",
        "why_it_might_help": "Age and experience can catch older overperformers, young breakouts, and limited-start QB risk.",
        "build_next": "Standardize age/experience features and test position-specific monotonic constraints.",
        "validation_gate": "Only use in position-specific models; age effects are not uniform across positions.",
        "spend": 0,
    },
    {
        "agent": "Matriculation Agent",
        "signal_family": "make_roster_snap_floor",
        "priority": 8,
        "free_source": "combine, draft picks, and NFL outcomes already in repo/nflverse",
        "candidate_features": "combine_participation, missing_testing_flags, athletic_z_completeness, drafted_vs_udfa floor signals",
        "why_it_might_help": "Combine data may help predict NFL matriculation even when it struggles to predict long-term success.",
        "build_next": "Separate floor/matriculation model from star/hit model so we do not optimize one target with the wrong features.",
        "validation_gate": "Evaluate separately with AUC/precision for played snaps or replacement-level career bucket.",
        "spend": 0,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    df = pd.DataFrame(SIGNALS).sort_values("priority")
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"signals": SIGNALS, "promotion_contract": "research only until residual, era-stability, sample-size, and rolling validation gates pass"}, indent=2))
    df.to_csv(out_csv, index=False)
    print(json.dumps({"status": "ok", "signals": len(df), "out_csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
