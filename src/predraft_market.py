"""Pre-draft consensus market scaffolding.

This script creates and validates the consensus-board file needed for true
pre-draft forecasting. The actual pre-draft model should use expected market
information available before draft night, not actual draft slot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline import ROOT

CONSENSUS_PATH = ROOT / "data" / "consensus" / "consensus_board.csv"
TEMPLATE_PATH = ROOT / "data" / "consensus" / "consensus_board_template.csv"
REQUIRED_COLUMNS = ["Year", "Player"]
MARKET_PROXY_COLUMNS = ["expected_pick", "consensus_rank", "mock_avg_pick"]
RECOMMENDED_COLUMNS = [
    "Year",
    "Player",
    "Pos",
    "College",
    "expected_pick",
    "consensus_rank",
    "mock_avg_pick",
    "mock_pick_std",
    "n_boards",
    "n_mocks",
    "nfl_com_grade",
    "source",
    "as_of_date",
]


def write_template(path: Path = TEMPLATE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=RECOMMENDED_COLUMNS).to_csv(path, index=False)
    return path


def check_consensus(path: Path = CONSENSUS_PATH) -> dict:
    write_template()
    if not path.exists():
        return {
            "ready": False,
            "path": str(path),
            "template": str(TEMPLATE_PATH),
            "reason": "consensus_board.csv not found",
            "required_columns": REQUIRED_COLUMNS,
            "market_proxy_columns_any_of": MARKET_PROXY_COLUMNS,
        }
    df = pd.read_csv(path)
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    proxy_present = [col for col in MARKET_PROXY_COLUMNS if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()]
    year_counts = df.groupby("Year").size().to_dict() if "Year" in df.columns else {}
    ready = not missing_required and bool(proxy_present)
    return {
        "ready": ready,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required": missing_required,
        "market_proxy_columns_present": proxy_present,
        "market_proxy_columns_any_of": MARKET_PROXY_COLUMNS,
        "year_counts": {str(k): int(v) for k, v in year_counts.items()},
        "reason": None if ready else "missing required columns or no usable expected-pick/consensus market proxy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=str, default=str(ROOT / "reports" / "predraft_market_status.json"))
    args = parser.parse_args()

    if args.write_template:
        path = write_template()
        print(f"Wrote template: {path}")
    status = check_consensus()
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
