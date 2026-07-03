"""Validate public site wording stays plain-language and slot-based."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / "src" / "template.html", ROOT / "docs" / "index.html", ROOT / "index.html"]
text = "\n".join(path.read_text(errors="ignore") for path in paths if path.exists())
errors: list[str] = []

banned_public_terms = [
    "APEX War Room",
    "Prospect Lens",
    "qb_model_greenlight",
    "qb_model_review",
    "Hold Grade",
    "Tier odds",
    "Lens score",
    "Very High",
]
for term in banned_public_terms:
    if term in text:
        errors.append(f"old/internal or exaggerated label still visible: {term}")

required_terms = [
    "Draft Projection Board",
    "Should Have Gone",
    "Slot Value",
    "Outcome Chances",
    "Bust Risk",
    "Star Chance",
    "Evidence",
]
for term in required_terms:
    if term not in text:
        errors.append(f"missing new plain-language label: {term}")

if "Model Pick" in text:
    errors.append("site should say Should Have Gone, not Model Pick")
if "range" in text.lower() and "slot" not in text:
    errors.append("site may still describe model output as a range instead of an exact slot")

if errors:
    print("FAIL")
    for error in errors:
        print(" -", error)
    sys.exit(1)
print("PASS: public site labels use should-have-gone slot language")
