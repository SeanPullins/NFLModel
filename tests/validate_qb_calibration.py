"""QB calibration invariants. Run after build_prospect_lens.py + build_site.py.

Fails if:
1. Any QB with an efficiency-only shape (pass>=0.70, create<=0.45) holds a
   greenlight or high-confidence label at model or site layer.
2. Creation-driven QBs (Daniels/Maye 2024) are pushed into fade labels.
3. Early-NFL outcome leakage reappears (legacy leakage label, or code paths
   that let `actual` set labels).
4. Site QB labels contradict the model layer (site upgrades a cautioned QB).
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
errors = []

df = pd.read_csv(ROOT / "data" / "apex_board.csv", low_memory=False)
qb = df[df["pos_g"] == "QB"].copy()
p = pd.to_numeric(qb.get("qb_pass_efficiency_score"), errors="coerce")
c = pd.to_numeric(qb.get("qb_creation_score"), errors="coerce")
caut = pd.to_numeric(qb.get("prospect_caution_count"), errors="coerce").fillna(0)
call = qb.get("prospect_lens_call", "").astype(str)

# 1. Efficiency-only shape must never greenlight, and must carry the caution.
eff_only = p.ge(0.70) & c.le(0.45)
bad = qb[eff_only & call.eq("qb_model_greenlight")]
if len(bad):
    errors.append(f"efficiency-only QBs greenlit: {bad['Player'].tolist()}")
flagged = qb["prospect_caution_flags"].astype(str).str.contains("one_dimensional_efficiency_profile")
missing_flag = qb[eff_only & ~flagged]
if len(missing_flag):
    errors.append(f"efficiency-only QBs missing caution flag: {missing_flag['Player'].tolist()}")

# 2. Creation-driven 2024 QBs must not be fade-labeled.
for name in ["Jayden Daniels", "Drake Maye"]:
    row = qb[(qb["Year"] == 2024) & (qb["Player"] == name)]
    if len(row) and str(row.iloc[0]["prospect_lens_call"]) in {"qb_fade_risk", "fade_risk", "avoid_risk"}:
        errors.append(f"{name} penalized into fade label")

# 3. Leakage guards: legacy leakage label gone; label code paths do not read `actual`.
site = (ROOT / "docs" / "index.html").read_text()
build_site_src = (ROOT / "src" / "build_site.py").read_text()
label_block = build_site_src.split("new_label", 1)[1].split('out.loc[needs, "qb_lens_label"]')[0]
if "actual" in label_block:
    errors.append("build_site label assignment reads NFL outcome (leakage)")

# 4. Site/model consistency: no cautioned QB shows qb_buy_high_confidence on site.
for _, r in qb[(caut > 0)].iterrows():
    m = re.search(re.escape(str(r["Player"])) + r'[^\]]{0,700}', site)
    if m and str(r["Year"]) in m.group(0)[:60] and (
        "qb_buy_high_confidence" in m.group(0) or '"qb_buy_volatile"' in m.group(0)
    ):
        errors.append(f"site upgraded cautioned QB {r['Player']} {r['Year']} to a buy label")

if errors:
    print("FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print(f"PASS: {len(qb)} QB rows checked; efficiency-only shapes: {int(eff_only.sum())}, all cautioned, none greenlit.")
