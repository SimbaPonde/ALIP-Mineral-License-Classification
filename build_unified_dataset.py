"""
ALIP – build the unified 170-document dataset (160 synthetic + 10 real Tier 1).

Writes data/unified_manifest.csv with repo-relative, forward-slash paths and a
fixed, stratified train / val / test split:

    train  109  synthetic only
    val     27  synthetic only (early stopping / model selection)
    test    34  24 synthetic (3 per class) + all 10 real Tier 1 documents

Real documents are only ever in the test split, so no model is trained or tuned
on them. Tier 2 / excluded real documents (see data/real_labels.csv) are written
to data/real_tier2_manifest.csv for optional out-of-scope analysis.

Usage:
    python build_unified_dataset.py
    python build_unified_dataset.py --synthetic ./output --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd

from alip_common import CLASS_LABELS, REPO_ROOT, UNIFIED_MANIFEST

TEST_PER_CLASS = 3
VAL_TOTAL = 27


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=os.path.join(REPO_ROOT, "output"))
    ap.add_argument("--real-labels", default=os.path.join(REPO_ROOT, "data", "real_labels.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ── synthetic ──────────────────────────────────────────────────────────
    syn = pd.read_csv(os.path.join(args.synthetic, "manifest.csv"))
    syn_rel = os.path.relpath(args.synthetic, REPO_ROOT).replace("\\", "/")
    syn = pd.DataFrame({
        "doc_id": syn["filename"].str.replace(".pdf", "", regex=False),
        "file_path": syn_rel + "/" + syn["file_path"].str.replace("\\", "/", regex=False),
        "label": syn["label"],
        "source": "synthetic",
        "jurisdiction": syn["jurisdiction"],
    })

    # stratified split: 3 test per class, val spread as evenly as possible
    val_per_class = {c: VAL_TOTAL // len(CLASS_LABELS) for c in CLASS_LABELS}
    for c in list(rng.permutation(CLASS_LABELS))[: VAL_TOTAL % len(CLASS_LABELS)]:
        val_per_class[c] += 1
    syn["split"] = "train"
    for c in CLASS_LABELS:
        idx = rng.permutation(syn.index[syn["label"] == c].to_numpy())
        syn.loc[idx[:TEST_PER_CLASS], "split"] = "test"
        syn.loc[idx[TEST_PER_CLASS:TEST_PER_CLASS + val_per_class[c]], "split"] = "val"

    # ── real ───────────────────────────────────────────────────────────────
    real = pd.read_csv(args.real_labels)
    t1 = real[real["tier"].astype(str) == "1"]
    bad = set(t1["label"]) - set(CLASS_LABELS)
    if bad:
        raise ValueError(f"Unknown Tier 1 labels in real_labels.csv: {bad}")
    real_t1 = pd.DataFrame({
        "doc_id": t1["file_path"].map(lambda p: "real_" + os.path.splitext(os.path.basename(p))[0]),
        "file_path": t1["file_path"],
        "label": t1["label"],
        "source": "real",
        "jurisdiction": t1["jurisdiction"],
        "split": "test",
    })

    df = pd.concat([syn, real_t1], ignore_index=True)
    missing = [p for p in df["file_path"] if not os.path.exists(os.path.join(REPO_ROOT, p))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing, e.g. {missing[:3]}")
    df.to_csv(UNIFIED_MANIFEST, index=False)
    real[real["tier"].astype(str) != "1"].to_csv(
        os.path.join(REPO_ROOT, "data", "real_tier2_manifest.csv"), index=False)

    print(f"Wrote {UNIFIED_MANIFEST}  ({len(df)} docs)")
    print(pd.crosstab([df["split"], df["source"]], df["label"], margins=True).to_string())


if __name__ == "__main__":
    main()
