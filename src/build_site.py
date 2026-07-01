from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "apex_board.csv"
TEMPLATE_PATH = ROOT / "src" / "template.html"
TARGETS = [ROOT / "index.html", ROOT / "docs" / "index.html"]

BASE_COLS = [
    "Year",
    "Player",
    "Pos",
    "pos_g",
    "College",
    "Pick",
    "Rnd",
    "CarAV",
    "y",
    "apex_score",
    "apex_raw",
    "exp_at_pick",
    "apex_edge",
    "raw_edge",
    "apex_conservative_025",
    "apex_conservative_075",
    "model_status",
]


def first_existing(df: pd.DataFrame, candidates: list[str], fallback: float | str | None = None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return fallback


df = pd.read_csv(DATA_PATH)
df["Pick"] = df["Pick"].where(df["Pick"] < 263)
df["College"] = df["College"].fillna("Unknown")

if "Rnd" not in df.columns:
    round_bins = [0, 32, 64, 100, 135, 176, 220, 262]
    df["Rnd"] = pd.cut(
        df["Pick"],
        bins=round_bins,
        labels=[1, 2, 3, 4, 5, 6, 7],
    ).astype("float")

# Site-facing score contract.
# Main score is the gate-passing APEX Conservative 0.50 candidate when present.
df["apex_raw"] = pd.to_numeric(df["apex"], errors="coerce")
df["apex_score"] = pd.to_numeric(first_existing(df, ["recommended_candidate_score", "apex_conservative_050", "apex"]), errors="coerce")
df["apex_edge"] = pd.to_numeric(first_existing(df, ["conservative_surplus_050", "surplus"]), errors="coerce")
df["raw_edge"] = pd.to_numeric(first_existing(df, ["surplus"], 0.0), errors="coerce")
if "apex_conservative_025" not in df.columns:
    df["apex_conservative_025"] = pd.to_numeric(df["exp_at_pick"], errors="coerce") + 0.25 * (df["apex_raw"] - pd.to_numeric(df["exp_at_pick"], errors="coerce"))
if "apex_conservative_075" not in df.columns:
    df["apex_conservative_075"] = pd.to_numeric(df["exp_at_pick"], errors="coerce") + 0.75 * (df["apex_raw"] - pd.to_numeric(df["exp_at_pick"], errors="coerce"))
if "model_status" not in df.columns:
    df["model_status"] = "apex_conservative_050_candidate"

for col in [
    "CarAV",
    "y",
    "apex_score",
    "apex_raw",
    "exp_at_pick",
    "apex_edge",
    "raw_edge",
    "apex_conservative_025",
    "apex_conservative_075",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

data = df[BASE_COLS].copy()
rows = data.astype(object).where(pd.notnull(data), None).values.tolist()
payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
html = TEMPLATE_PATH.read_text().replace("__DATA__", payload)

for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    out = html
    if target.parent.name == "docs":
        out = out.replace('href="docs/', 'href="')
    target.write_text(out)

print("rows:", len(rows), "size:", len(html) // 1024, "KB")
print("main_score: apex_conservative_050 via apex_score")
