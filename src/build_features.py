"""Build an enriched modeling table from optional production/consensus files.

This script is intentionally conservative: missing optional files are reported,
not treated as errors. Drop CSVs into data/production/ or data/consensus/, run
this script, then use data/model_features.csv in experiment scripts.

Required merge keys in optional files:
    Year, Player

Recommended optional ID columns:
    Pos, College, Pick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_registry import (
    CONSENSUS_DIR,
    ENRICHED_FEATURE_FILE,
    FEATURE_COVERAGE_REPORT,
    FEATURE_FILE_SPECS,
    KEY_COLUMNS,
    OPTIONAL_ID_COLUMNS,
)
from pipeline import ROOT, load_dataset, norm


def make_key(df: pd.DataFrame) -> pd.Series:
    return df["Player"].map(norm) + "_" + pd.to_numeric(df["Year"], errors="coerce").astype("Int64").astype(str)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for col in out.columns:
        clean = str(col).strip()
        if clean != col:
            rename[col] = clean
    out = out.rename(columns=rename)
    aliases = {
        "player": "Player",
        "name": "Player",
        "draft_year": "Year",
        "year": "Year",
        "college_school": "College",
        "school": "College",
        "position": "Pos",
        "pos": "Pos",
    }
    return out.rename(columns={c: aliases.get(c, c) for c in out.columns})


def read_optional_file(path: Path, feature_columns: tuple[str, ...]) -> tuple[pd.DataFrame | None, dict]:
    meta = {
        "path": str(path),
        "exists": path.exists(),
        "rows": 0,
        "features_present": [],
        "features_missing": list(feature_columns),
        "usable": False,
    }
    if not path.exists():
        return None, meta

    raw = normalize_columns(pd.read_csv(path))
    missing_keys = [c for c in KEY_COLUMNS if c not in raw.columns]
    if missing_keys:
        meta["error"] = f"Missing required key columns: {missing_keys}"
        return None, meta

    raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce")
    raw = raw[raw["Year"].notna() & raw["Player"].notna()].copy()
    raw["Year"] = raw["Year"].astype(int)
    raw["key"] = make_key(raw)

    feature_present = [c for c in feature_columns if c in raw.columns]
    keep = ["key", *[c for c in OPTIONAL_ID_COLUMNS if c in raw.columns], *feature_present]
    out = raw[keep].drop_duplicates("key").copy()
    for col in feature_present:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    meta.update(
        {
            "rows": int(len(out)),
            "features_present": feature_present,
            "features_missing": [c for c in feature_columns if c not in raw.columns],
            "usable": bool(feature_present),
        }
    )
    return out, meta


def merge_optional_features(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = base.copy()
    out["key"] = make_key(out)
    report = {"files": {}, "feature_coverage": {}, "rows": int(len(out))}

    for spec in FEATURE_FILE_SPECS:
        optional, meta = read_optional_file(spec.path, spec.feature_columns)
        report["files"][spec.name] = meta
        if optional is None or not meta.get("usable"):
            continue
        feature_cols = meta["features_present"]
        before_cols = set(out.columns)
        out = out.merge(optional[["key", *feature_cols]], on="key", how="left")
        added = [c for c in out.columns if c not in before_cols]
        for col in added:
            coverage = float(out[col].notna().mean()) if len(out) else np.nan
            report["feature_coverage"][col] = {
                "coverage": coverage,
                "non_null": int(out[col].notna().sum()),
            }

    return out.drop(columns=["key"]), report


def write_templates(template_dir: Path) -> None:
    template_dir.mkdir(parents=True, exist_ok=True)
    for spec in FEATURE_FILE_SPECS:
        if spec.path.exists():
            continue
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        columns = [*KEY_COLUMNS, "Pos", "College", *spec.feature_columns]
        pd.DataFrame(columns=columns).to_csv(spec.path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(ENRICHED_FEATURE_FILE))
    parser.add_argument("--report", type=str, default=str(FEATURE_COVERAGE_REPORT))
    parser.add_argument("--write-templates", action="store_true")
    args = parser.parse_args()

    if args.write_templates:
        write_templates(ROOT / "data")

    base = load_dataset(data_dir=args.data_dir)
    enriched, report = merge_optional_features(base)

    out_path = Path(args.out)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    enriched.round(6).to_csv(out_path, index=False)
    report_path.write_text(json.dumps(report, indent=2))

    loaded = [name for name, meta in report["files"].items() if meta.get("usable")]
    missing = [name for name, meta in report["files"].items() if not meta.get("exists")]
    print(f"Wrote enriched features: {out_path}")
    print(f"Wrote feature coverage report: {report_path}")
    print(f"Loaded optional files: {loaded or 'none'}")
    print(f"Missing optional files: {missing or 'none'}")
    if args.write_templates:
        print("Template CSVs were created for missing optional feature files.")


if __name__ == "__main__":
    main()
