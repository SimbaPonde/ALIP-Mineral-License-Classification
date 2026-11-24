"""
ALIP – Aquila Land Intelligence Platform
Tier 1 Baseline Pipeline: TF-IDF + Logistic Regression / SVM
=============================================================
Steps:
  1. Extract text from PDFs (via pdfplumber)
  2. Preprocess text (clean, tokenise, remove stopwords, lemmatise)
  3. Build TF-IDF feature matrix
  4. Train and evaluate Logistic Regression and SVM classifiers
     using stratified k-fold cross-validation (k=5)
  5. Report accuracy, precision, recall, macro-averaged F1-score,
     confusion matrix, and per-class breakdown
  6. Save all results to results/baseline_results.csv and plots/

Usage:
  python baseline_pipeline.py
  python baseline_pipeline.py --data ./output --results ./results/synthetic_only_cv
  python baseline_pipeline.py --data ./output --folds 10
"""

import os
import re
import csv
import json
import warnings
import argparse
import time

import numpy as np
import pandas as pd
import pdfplumber
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── NLTK with graceful fallback ───────────────────────────────────────────────

FALLBACK_STOPWORDS = {
    "a","about","above","after","again","against","all","also","am","an","and",
    "any","are","as","at","be","because","been","before","being","below","between",
    "both","but","by","can","did","do","does","doing","down","during","each","few",
    "for","from","further","get","had","has","have","having","he","her","here",
    "hers","herself","him","himself","his","how","i","if","in","into","is","it",
    "its","itself","just","me","more","most","my","myself","no","nor","not","now",
    "of","off","on","once","only","or","other","our","ours","out","over","own",
    "s","same","she","should","so","some","such","t","than","that","the","their",
    "theirs","them","themselves","then","there","these","they","this","those",
    "through","to","too","under","until","up","us","very","was","we","were","what",
    "when","where","which","while","who","whom","why","will","with","would","you",
    "your","yours","yourself","yourselves","shall","may","must","hereby","herein",
    "thereof","thereto","hereunder","hereinafter","pursuant","accordance","said",
    "made","upon","within","without","whether","however","therefore","whereas",
    "provided","subject","including","following","above","mentioned","attached",
}

def get_stopwords():
    try:
        import nltk
        nltk.data.path.insert(0, os.path.join(os.path.dirname(__file__), "nltk_data"))
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
        sw.update(FALLBACK_STOPWORDS)
        return sw
    except Exception:
        return FALLBACK_STOPWORDS

def get_lemmatizer():
    try:
        import nltk
        nltk.data.path.insert(0, os.path.join(os.path.dirname(__file__), "nltk_data"))
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("test")   # trigger lazy load — raises if wordnet missing
        return lem
    except Exception:
        return None

STOPWORDS = get_stopwords()
LEMMATIZER = get_lemmatizer()

# ── CLASS LABELS ──────────────────────────────────────────────────────────────

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
    "exploration_license":               "Exploration License",
    "prospecting_permit":                "Prospecting Permit",
    "mining_lease":                      "Mining Lease",
    "small_scale_mining_permit":         "Small-Scale Mining Permit",
    "new_license_application":           "New License Application",
    "renewal_application":               "Renewal Application",
    "environmental_impact_assessment":   "Environmental Impact Assessment",
    "environmental_compliance_certificate": "Env. Compliance Certificate",
}

# ── STEP 1: PDF TEXT EXTRACTION ───────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    except Exception as e:
        print(f"  WARNING: Could not extract text from {pdf_path}: {e}")
    return " ".join(text_parts)


def load_dataset(data_dir: str, manifest_path: str = None) -> pd.DataFrame:
    """
    Load all documents from the output directory.
    Uses manifest.csv if present, otherwise walks the directory structure.
    Returns a DataFrame with columns: [file_path, label, text]
    """
    print("\n[Step 1] Loading dataset and extracting text from PDFs...")

    records = []

    # Prefer manifest if available
    if manifest_path and os.path.exists(manifest_path):
        print(f"  Reading manifest: {manifest_path}")
        df_manifest = pd.read_csv(manifest_path)
        for _, row in df_manifest.iterrows():
            abs_path = os.path.join(data_dir, row["file_path"])
            if os.path.exists(abs_path):
                records.append({
                    "file_path": abs_path,
                    "label": row["label"],
                    "jurisdiction": row.get("jurisdiction", "unknown"),
                    "filename": row["filename"],
                })
    else:
        # Walk directory structure: output/<class_name>/*.pdf
        print(f"  No manifest found. Scanning directory: {data_dir}")
        for class_name in CLASS_LABELS:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"  WARNING: Class directory not found: {class_dir}")
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(".pdf"):
                    records.append({
                        "file_path": os.path.join(class_dir, fname),
                        "label": class_name,
                        "jurisdiction": "unknown",
                        "filename": fname,
                    })

    if not records:
        raise FileNotFoundError(
            f"No PDF documents found in {data_dir}. "
            "Make sure generate_synthetic_docs.py has been run first."
        )

    df = pd.DataFrame(records)
    print(f"  Found {len(df)} documents across {df['label'].nunique()} classes.")

    # Extract text
    texts = []
    for i, row in df.iterrows():
        text = extract_text_from_pdf(row["file_path"])
        texts.append(text)
        if (i + 1) % 20 == 0 or (i + 1) == len(df):
            print(f"  Extracted text: {i+1}/{len(df)} documents", end="\r")

    df["text"] = texts
    df["text_length"] = df["text"].str.len()
    empty = (df["text_length"] == 0).sum()
    print(f"\n  Text extraction complete. Empty documents: {empty}")
    print(f"  Avg text length: {df['text_length'].mean():.0f} chars | "
          f"Min: {df['text_length'].min()} | Max: {df['text_length'].max()}")
    return df


# ── STEP 2: TEXT PREPROCESSING ───────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalise document text:
    1. Lowercase
    2. Remove numbers, punctuation, and special characters
    3. Tokenise by whitespace
    4. Remove stopwords
    5. Lemmatise (if NLTK available) or strip suffixes
    6. Remove very short tokens
    """
    if not text or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)

    # Remove numbers (keep words containing letters)
    text = re.sub(r"\b\d+[\d.,]*\b", " ", text)

    # Remove punctuation and special characters, keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenise
    tokens = text.split()

    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    # Lemmatise or fallback to simple suffix stripping
    if LEMMATIZER:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    else:
        # Simple suffix stripping as fallback
        cleaned = []
        for t in tokens:
            if t.endswith("ations"):  t = t[:-6] + "ate"
            elif t.endswith("ation"): t = t[:-5] + "ate"
            elif t.endswith("ings"):  t = t[:-4]
            elif t.endswith("ing"):   t = t[:-3]
            elif t.endswith("ies"):   t = t[:-3] + "y"
            elif t.endswith("ness"):  t = t[:-4]
            elif t.endswith("ment"):  t = t[:-4]
            elif t.endswith("able"):  t = t[:-4]
            elif t.endswith("tion"):  t = t[:-4]
            elif t.endswith("ed") and len(t) > 4: t = t[:-2]
            elif t.endswith("es") and len(t) > 4: t = t[:-2]
            elif t.endswith("s") and len(t) > 4:  t = t[:-1]
            cleaned.append(t)
        tokens = cleaned

    return " ".join(tokens)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 2] Preprocessing text...")
    df = df.copy()
    df["text_clean"] = df["text"].apply(preprocess_text)
    empty_after = (df["text_clean"].str.len() == 0).sum()
    avg_tokens = df["text_clean"].str.split().str.len().mean()
    print(f"  Preprocessing complete.")
    print(f"  Avg token count after preprocessing: {avg_tokens:.0f}")
    print(f"  Documents with no tokens after preprocessing: {empty_after}")
    return df


# ── STEP 3 & 4: TFIDF + CLASSIFIERS ─────────────────────────────────────────

def build_pipelines():
    """
    Returns a dict of named sklearn Pipelines.
    Each pipeline: TF-IDF vectoriser → classifier
    """
    tfidf_params = dict(
        ngram_range=(1, 2),       # unigrams + bigrams
        max_features=5000,        # top 5,000 features
        sublinear_tf=True,        # apply log(TF) scaling
        min_df=2,                 # ignore terms in fewer than 2 documents
        strip_accents="unicode",
    )

    pipelines = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "TF-IDF + SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
            )),
        ]),
    }
    return pipelines


def run_cross_validation(pipelines: dict, X, y, n_folds: int = 5):
    """
    Run stratified k-fold cross-validation for each pipeline.
    Returns dict of results per model.
    """
    print(f"\n[Step 3 & 4] Training and evaluating models "
          f"({n_folds}-fold stratified cross-validation)...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

    all_results = {}

    for model_name, pipeline in pipelines.items():
        print(f"\n  ── {model_name} ──")
        t0 = time.time()

        cv_scores = cross_validate(
            pipeline, X, y,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        elapsed = time.time() - t0

        results = {
            "model":             model_name,
            "accuracy_mean":     cv_scores["test_accuracy"].mean(),
            "accuracy_std":      cv_scores["test_accuracy"].std(),
            "f1_macro_mean":     cv_scores["test_f1_macro"].mean(),
            "f1_macro_std":      cv_scores["test_f1_macro"].std(),
            "precision_mean":    cv_scores["test_precision_macro"].mean(),
            "precision_std":     cv_scores["test_precision_macro"].std(),
            "recall_mean":       cv_scores["test_recall_macro"].mean(),
            "recall_std":        cv_scores["test_recall_macro"].std(),
            "train_time_sec":    round(elapsed, 2),
            "cv_raw":            cv_scores,
        }

        print(f"  Accuracy:         {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"  Macro F1-score:   {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
        print(f"  Macro Precision:  {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Macro Recall:     {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"  Training time:    {elapsed:.2f}s")

        all_results[model_name] = results

    return all_results


def get_per_class_report(pipeline, X, y, label_names, n_folds=5):
    """
    Re-run cross-validation collecting per-fold predictions
    to build a full per-class classification report.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train = [X[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        all_true.extend(y_test)
        all_pred.extend(y_pred)

    report = classification_report(
        all_true, all_pred,
        target_names=label_names,
        output_dict=True,
    )
    cm = confusion_matrix(all_true, all_pred)
    return report, cm, all_true, all_pred


# ── STEP 5: VISUALISATIONS ───────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, model_name, output_dir):
    short_names = [CLASS_DISPLAY.get(c, c) for c in class_names]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
        ax=ax, linewidths=0.5, linecolor="grey",
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "_").replace("+", "plus")
    path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_per_class_f1(report, class_names, model_name, output_dir):
    short_names = [CLASS_DISPLAY.get(c, c) for c in class_names]
    f1_scores = [report[c]["f1-score"] for c in class_names]

    colors_list = ["#1F4E79" if f >= 0.7 else "#2E75B6" if f >= 0.5
                   else "#F4A300" for f in f1_scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(short_names, f1_scores, color=colors_list, edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_title(f"Per-Class F1-Score — {model_name}", fontsize=13, fontweight="bold")
    ax.axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="0.70 threshold")
    ax.axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="0.50 threshold")

    for bar, val in zip(bars, f1_scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.legend(fontsize=9)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "_").replace("+", "plus")
    path = os.path.join(output_dir, f"per_class_f1_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_model_comparison(all_results, output_dir):
    models = list(all_results.keys())
    metrics = {
        "Macro F1":   [all_results[m]["f1_macro_mean"]  for m in models],
        "Accuracy":   [all_results[m]["accuracy_mean"]  for m in models],
        "Precision":  [all_results[m]["precision_mean"] for m in models],
        "Recall":     [all_results[m]["recall_mean"]    for m in models],
    }
    errs = {
        "Macro F1":   [all_results[m]["f1_macro_std"]  for m in models],
        "Accuracy":   [all_results[m]["accuracy_std"]  for m in models],
        "Precision":  [all_results[m]["precision_std"] for m in models],
        "Recall":     [all_results[m]["recall_std"]    for m in models],
    }

    x = np.arange(len(models))
    width = 0.2
    palette = ["#1F4E79", "#2E75B6", "#F4A300", "#E05C2A"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width,
               yerr=errs[metric], capsize=4,
               label=metric, color=palette[i], alpha=0.88)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace("TF-IDF + ", "") for m in models], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Tier 1 Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_class_distribution(df, output_dir):
    counts = df["label"].value_counts()
    short = [CLASS_DISPLAY.get(c, c) for c in counts.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(short, counts.values, color="#1F4E79", edgecolor="white", alpha=0.85)
    ax.set_ylabel("Number of Documents", fontsize=11)
    ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── STEP 6: SAVE RESULTS ─────────────────────────────────────────────────────

def save_results(all_results, per_class_reports, output_dir):
    """Save summary CSV and detailed JSON report."""

    # Summary CSV
    summary_rows = []
    for model_name, res in all_results.items():
        summary_rows.append({
            "model":          model_name,
            "accuracy":       round(res["accuracy_mean"],  4),
            "accuracy_std":   round(res["accuracy_std"],   4),
            "f1_macro":       round(res["f1_macro_mean"],  4),
            "f1_macro_std":   round(res["f1_macro_std"],   4),
            "precision":      round(res["precision_mean"], 4),
            "precision_std":  round(res["precision_std"],  4),
            "recall":         round(res["recall_mean"],    4),
            "recall_std":     round(res["recall_std"],     4),
            "train_time_sec": res["train_time_sec"],
        })
    csv_path = os.path.join(output_dir, "baseline_results.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  Saved summary: {csv_path}")

    # Per-class detailed JSON
    detail = {}
    for model_name, report in per_class_reports.items():
        detail[model_name] = {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1_score":  round(report[cls]["f1-score"],  4),
                "support":   report[cls]["support"],
            }
            for cls in CLASS_LABELS if cls in report
        }
        detail[model_name]["macro avg"] = report.get("macro avg", {})
        detail[model_name]["weighted avg"] = report.get("weighted avg", {})

    json_path = os.path.join(output_dir, "baseline_per_class.json")
    with open(json_path, "w") as f:
        json.dump(detail, f, indent=2)
    print(f"  Saved per-class detail: {json_path}")

    return csv_path, json_path


def print_summary_table(all_results):
    print("\n" + "="*65)
    print("  TIER 1 BASELINE — FINAL RESULTS SUMMARY")
    print("="*65)
    print(f"  {'Model':<38} {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print("  " + "-"*61)
    for model_name, res in all_results.items():
        short = model_name.replace("TF-IDF + ", "")
        print(f"  {short:<38} "
              f"{res['accuracy_mean']:>6.4f}  "
              f"{res['f1_macro_mean']:>6.4f}  "
              f"{res['precision_mean']:>6.4f}  "
              f"{res['recall_mean']:>6.4f}")
    print("="*65)
    print("  Primary metric: Macro-averaged F1-score")
    print("  (Macro F1 weights all classes equally regardless of support)")
    print("="*65 + "\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ALIP Tier 1 Baseline Pipeline")
    parser.add_argument("--data",    type=str, default="./output",
                        help="Directory containing generated documents (default: ./output)")
    parser.add_argument("--results", type=str, default="./results/synthetic_only_cv",
                        help="Directory for saving results (default: ./results)")
    parser.add_argument("--folds",   type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    args = parser.parse_args()

    results_dir = args.results
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    manifest = os.path.join(args.data, "manifest.csv")

    print("\n" + "="*65)
    print("  ALIP — Tier 1 Baseline Pipeline")
    print(f"  Data:    {args.data}")
    print(f"  Results: {results_dir}")
    print(f"  Folds:   {args.folds}")
    print("="*65)

    # ── 1. Load data ──────────────────────────────────────────────
    df = load_dataset(args.data, manifest)

    # ── 2. Preprocess ────────────────────────────────────────────
    df = preprocess_dataset(df)

    # ── Plot class distribution ──────────────────────────────────
    print("\n[Plotting] Class distribution...")
    plot_class_distribution(df, plots_dir)

    # ── 3. Prepare features and labels ──────────────────────────
    X = df["text_clean"].tolist()
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    y = le.transform(df["label"])
    label_names = list(le.classes_)
    print(f"\n[Step 3] Features prepared: {len(X)} documents, {len(label_names)} classes")

    # ── 4. Build pipelines ───────────────────────────────────────
    pipelines = build_pipelines()

    # ── 5. Cross-validation ──────────────────────────────────────
    all_results = run_cross_validation(pipelines, X, y, n_folds=args.folds)

    # ── 6. Per-class reports and confusion matrices ──────────────
    print("\n[Step 5] Generating per-class reports and confusion matrices...")
    per_class_reports = {}
    for model_name, pipeline in pipelines.items():
        print(f"\n  ── {model_name} ──")
        report, cm, y_true, y_pred = get_per_class_report(
            pipeline, X, y, label_names, n_folds=args.folds)
        per_class_reports[model_name] = report

        # Print per-class table
        print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'n':>5}")
        print("  " + "-"*65)
        for cls in label_names:
            if cls in report:
                r = report[cls]
                disp = CLASS_DISPLAY.get(cls, cls)
                print(f"  {disp:<45} "
                      f"{r['precision']:>6.3f} "
                      f"{r['recall']:>6.3f} "
                      f"{r['f1-score']:>6.3f} "
                      f"{int(r['support']):>5}")
        macro = report.get("macro avg", {})
        print("  " + "-"*65)
        print(f"  {'MACRO AVERAGE':<45} "
              f"{macro.get('precision', 0):>6.3f} "
              f"{macro.get('recall', 0):>6.3f} "
              f"{macro.get('f1-score', 0):>6.3f}")

        # Plots
        plot_confusion_matrix(cm, label_names, model_name, plots_dir)
        plot_per_class_f1(report, label_names, model_name, plots_dir)

    # ── 7. Model comparison plot ─────────────────────────────────
    print("\n[Plotting] Model comparison chart...")
    plot_model_comparison(all_results, plots_dir)

    # ── 8. Save results ──────────────────────────────────────────
    print("\n[Step 6] Saving results...")
    save_results(all_results, per_class_reports, results_dir)

    # ── 9. Final summary ─────────────────────────────────────────
    print_summary_table(all_results)

    best = max(all_results.items(), key=lambda x: x[1]["f1_macro_mean"])
    print(f"  Best model: {best[0]}")
    print(f"  Best macro F1: {best[1]['f1_macro_mean']:.4f}")
    print(f"\n  Results saved to: {results_dir}/")
    print(f"  Plots saved to:   {plots_dir}/\n")


if __name__ == "__main__":
    main()
