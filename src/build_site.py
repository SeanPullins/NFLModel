from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "apex_board.csv"
TEMPLATE_PATH = ROOT / "src" / "template.html"
TARGETS = [ROOT / "index.html", ROOT / "docs" / "index.html"]

cols = [
    "Year",
    "Player",
    "Pos",
    "pos_g",
    "College",
    "Pick",
    "Rnd",
    "CarAV",
    "y",
    "apex",
    "exp_at_pick",
    "talent_resid",
    "surplus",
]

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

data = df[cols].copy()
for col in ["CarAV", "y", "apex", "exp_at_pick", "talent_resid", "surplus"]:
    data[col] = data[col].round(4)

rows = data.astype(object).where(pd.notnull(data), None).values.tolist()
payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
html = TEMPLATE_PATH.read_text().replace("__DATA__", payload)

for target in TARGETS:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html)

print("rows:", len(rows), "size:", len(html) // 1024, "KB")
