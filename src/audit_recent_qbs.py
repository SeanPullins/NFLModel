"""Recent QB audit: pre-draft projection vs market vs live evidence for the
named QB list. Output feeds the dashboard-facing corrected labels."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "Caleb Williams", "Jayden Daniels", "Drake Maye", "J.J. McCarthy", "Bo Nix",
    "Michael Penix", "Spencer Rattler", "C.J. Stroud", "Bryce Young",
    "Anthony Richardson", "Will Levis", "Hendon Hooker", "Kenny Pickett",
    "Zach Wilson", "Trey Lance", "Justin Fields", "Mac Jones",
]

df = pd.read_csv(ROOT / "data" / "apex_board.csv", low_memory=False)
qb = df[(df["pos_g"] == "QB") & (df["Year"] >= 2021)].copy()
rows = []
for name in NAMES:
    m = qb[
        qb["Player"].str.contains(name.split()[-1], na=False)
        & qb["Player"].str.contains(name.split()[0].rstrip("."), na=False, regex=False)
    ]
    if m.empty:
        m = qb[qb["Player"] == name]
    if m.empty:
        rows.append({"player": name, "status": "not_found"})
        continue
    r = m.iloc[0]
    y = pd.to_numeric(pd.Series([r.get("y")]), errors="coerce").iloc[0]
    year = int(r["Year"])
    if year < 2024:
        phase = "historical"
    elif pd.notna(y):
        phase = "partial_live"
    else:
        phase = "projection_only"
    live = (
        "no NFL outcome yet" if pd.isna(y)
        else f"partial live outcome {y:.2f} " + ("(confirming)" if y >= 0.55 else "(below projection)" if y < 0.45 else "(mixed)")
    )
    rows.append({
        "player": r["Player"],
        "year": year,
        "pick": r.get("Pick"),
        "pre_draft_projection": r.get("prospect_lens_score"),
        "market_expectation": r.get("exp_at_pick"),
        "live_evidence": live,
        "phase": phase,
        "reasons": r.get("qb_lens_reasons", ""),
        "warnings": f'{r.get("qb_lens_warning", "")};{r.get("prospect_caution_flags", "")}',
        "dashboard_label": r.get("qb_lens_label") or r.get("prospect_lens_call"),
    })
out = pd.DataFrame(rows)
out.to_csv(ROOT / "reports" / "qb_recent_forecast.csv", index=False)
print(out[["player", "year", "phase", "dashboard_label", "live_evidence"]].to_string(index=False))
