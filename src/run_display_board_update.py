"""Run the clean display-board update sequence.

This wrapper exists so agents or local runs can apply the UX/odds fix without
remembering the command order.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "src/build_prospect_lens.py", "--board", "data/apex_board.csv", "--cfbd", "data/production/cfbd_production.csv", "--report", "reports/prospect_lens_report.json", "--recent", "reports/prospect_lens_recent.csv"],
    [sys.executable, "src/calibrate_outcome_odds.py", "--board", "data/apex_board.csv", "--report", "reports/outcome_odds_calibration_report.json", "--curve", "reports/outcome_odds_calibration.csv"],
    [sys.executable, "src/build_site.py"],
    [sys.executable, "tests/validate_qb_calibration.py"],
    [sys.executable, "tests/validate_outcome_odds.py"],
    [sys.executable, "tests/validate_site_labels.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("$", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
