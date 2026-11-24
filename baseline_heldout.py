"""
ALIP – Tier 1 baseline on the fixed held-out split.

Trains TF-IDF + Logistic Regression and TF-IDF + LinearSVC on the synthetic
`train` split and predicts the `test` split (24 synthetic + 10 real). Writes
prediction files in the shared format for compare_models.py.

(baseline_pipeline.py still provides the original 5-fold CV analysis.)

Usage:
    python baseline_heldout.py
    python baseline_heldout.py --mask-class-names     # label-leakage ablation
    python baseline_heldout.py --train-splits train val
"""

import argparse
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from alip_common import abspath, extract_text, load_manifest, mask_class_names, save_predictions
from baseline_pipeline import preprocess_text

TFIDF = dict(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2,
             strip_accents="unicode")


def models():
    return {
        "TF-IDF + Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "TF-IDF + LinearSVC": LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }


def texts(df, mask):
    out = []
    for p in df["file_path"]:
        t = extract_text(abspath(p))
        out.append(preprocess_text(mask_class_names(t) if mask else t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-class-names", action="store_true",
                    help="remove phrases that name a class before vectorising (leakage ablation)")
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    args = ap.parse_args()

    df = load_manifest("all")
    tr = df[df["split"].isin(args.train_splits)]
    te = df[df["split"] == "test"]
    print(f"train={len(tr)}  test={len(te)} (real={int((te.source == 'real').sum())})"
          f"  mask_class_names={args.mask_class_names}")

    Xtr, Xte = texts(tr, args.mask_class_names), texts(te, args.mask_class_names)
    suffix = " [masked]" if args.mask_class_names else ""
    for name, clf in models().items():
        t0 = time.time()
        pipe = Pipeline([("tfidf", TfidfVectorizer(**TFIDF)), ("clf", clf)]).fit(Xtr, tr["label"])
        pred = pipe.predict(Xte)
        rows = [{"doc_id": d, "true": y, "pred": p, "source": s}
                for d, y, p, s in zip(te["doc_id"], te["label"], pred, te["source"])]
        acc = sum(r["true"] == r["pred"] for r in rows) / len(rows)
        path = save_predictions(name + suffix, 1, "test", rows,
                                {"train_splits": args.train_splits, "tfidf": TFIDF,
                                 "mask_class_names": args.mask_class_names,
                                 "train_time_sec": round(time.time() - t0, 2)})
        print(f"{name + suffix:40s} acc={acc:.3f} -> {path}")


if __name__ == "__main__":
    main()
