"""
ALIP – shared constants and helpers used by every tier.

Prediction file format (results/predictions/<model_slug>.json), shared by all models
so compare_models.py can evaluate them identically:

{
  "model": "TF-IDF + Logistic Regression",
  "tier": 1,
  "split": "test",
  "meta": {...},
  "predictions": [
      {"doc_id": "...", "true": "<label>", "pred": "<label>", "source": "real|synthetic"},
      ...
  ]
}
"""

import json
import os
import re

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_MANIFEST = os.path.join(REPO_ROOT, "data", "unified_manifest.csv")
PRED_DIR = os.path.join(REPO_ROOT, "results", "predictions")

CLASS_LABELS = [
    "exploration_license",
    "prospecting_permit",
    "mining_lease",
    "small_scale_mining_permit",
    "new_license_application",
    "renewal_application",
    "environmental_impact_assessment",
    "environmental_compliance_certificate",
]

CLASS_DISPLAY = {
    "exploration_license": "Exploration License",
    "prospecting_permit": "Prospecting Permit",
    "mining_lease": "Mining Lease",
    "small_scale_mining_permit": "Small-Scale Mining Permit",
    "new_license_application": "New License Application",
    "renewal_application": "Renewal Application",
    "environmental_impact_assessment": "Environmental Impact Assessment",
    "environmental_compliance_certificate": "Environmental Compliance Certificate",
}

# Short descriptions used in the LLM prompt (Tier 3).
CLASS_DESCRIPTIONS = {
    "exploration_license": "An issued licence granting the holder the right to carry out geological/geophysical exploration over a defined area.",
    "prospecting_permit": "An issued permit or licence allowing initial prospecting to establish whether minerals are present.",
    "mining_lease": "An issued lease or licence granting rights to extract minerals at commercial scale.",
    "small_scale_mining_permit": "A permit or licence for artisanal or small-scale mining operations (including applications specifically for such permits).",
    "new_license_application": "An application form requesting the issue of a new mineral licence or permit.",
    "renewal_application": "An application to renew or extend an existing mineral licence before it expires.",
    "environmental_impact_assessment": "An environmental impact assessment report studying the environmental effects of a proposed mining project.",
    "environmental_compliance_certificate": "A certificate confirming that a project has satisfied its EIA / environmental management requirements.",
}

# Phrases that name a class outright. Used for the label-leakage ablation
# (--mask-class-names): every occurrence is removed before vectorising.
CLASS_NAME_PATTERNS = [
    r"exploration\s+licen[cs]e", r"prospecting\s+(permit|licen[cs]e)",
    r"mining\s+(lease|licen[cs]e)", r"small[\s-]*scale\s+mining\s+(permit|licen[cs]e)",
    r"artisanal\s+mining\s+licen[cs]e", r"application\s+for", r"renewal",
    r"environmental\s+impact\s+assessment", r"environmental\s+compliance\s+certificate",
    r"\beia\b", r"\bpermit\b", r"\blicen[cs]e\b", r"\blease\b", r"\bcertificate\b",
]


def mask_class_names(text: str) -> str:
    for p in CLASS_NAME_PATTERNS:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return text


def abspath(rel_path: str) -> str:
    """Manifest paths are repo-relative with forward slashes (works on Windows and Colab)."""
    return os.path.join(REPO_ROOT, *rel_path.replace("\\", "/").split("/"))


def extract_text(pdf_path: str, max_pages: int = None) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages is not None and i >= max_pages:
                break
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def load_manifest(split: str = None, path: str = UNIFIED_MANIFEST) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run build_unified_dataset.py first.")
    df = pd.read_csv(path)
    if split and split != "all":
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


def save_predictions(model: str, tier: int, split: str, rows: list, meta: dict = None,
                     out_dir: str = PRED_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    path = os.path.join(out_dir, f"{slug}__{split}.json")
    with open(path, "w") as f:
        json.dump({"model": model, "tier": tier, "split": split, "meta": meta or {},
                   "predictions": rows}, f, indent=2)
    return path
