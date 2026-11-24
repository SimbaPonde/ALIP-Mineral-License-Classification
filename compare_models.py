"""
ALIP – compare every model on identical documents (rebuilt).

Reads results/predictions/*__<split>.json and reports, for each model:
  * macro F1 / accuracy / macro precision / macro recall on the whole split
  * accuracy on the real documents only (the generalisation signal)
  * per-class F1
  * pairwise exact McNemar tests on the same documents
Writes results/comparison/{summary.csv, per_class_f1.csv, mcnemar.csv, comparison.md}
and plots.

Usage:
    python compare_models.py                 # held-out test split
    python compare_models.py --split all
"""

import argparse
import glob
import itertools
import json
import os

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from alip_common import CLASS_DISPLAY, CLASS_LABELS, PRED_DIR, REPO_ROOT

OUT = os.path.join(REPO_ROOT, "results", "comparison")


def mcnemar_exact(a_correct, b_correct):
    b = sum(x and not y for x, y in zip(a_correct, b_correct))   # A right, B wrong
    c = sum(y and not x for x, y in zip(a_correct, b_correct))   # B right, A wrong
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(PRED_DIR, f"*__{args.split}.json")))
    if not files:
        raise SystemExit(f"No prediction files for split '{args.split}' in {PRED_DIR}")
    runs = [json.load(open(f)) for f in files]
    runs.sort(key=lambda r: (r["tier"], r["model"]))

    # align on the documents every model predicted
    common = set.intersection(*[{p["doc_id"] for p in r["predictions"]} for r in runs])
    summary, per_class, correct = [], {}, {}
    for r in runs:
        preds = sorted((p for p in r["predictions"] if p["doc_id"] in common), key=lambda p: p["doc_id"])
        y = [p["true"] for p in preds]
        yh = [p["pred"] for p in preds]
        real = [p for p in preds if p["source"] == "real"]
        kw = dict(labels=CLASS_LABELS, average="macro", zero_division=0)
        summary.append({
            "model": r["model"], "tier": r["tier"], "n": len(preds),
            "macro_f1": f1_score(y, yh, **kw), "accuracy": accuracy_score(y, yh),
            "macro_precision": precision_score(y, yh, **kw), "macro_recall": recall_score(y, yh, **kw),
            "correct": sum(a == b for a, b in zip(y, yh)),
            "real_correct": f"{sum(p['true'] == p['pred'] for p in real)}/{len(real)}",
        })
        f1s = f1_score(y, yh, labels=CLASS_LABELS, average=None, zero_division=0)
        per_class[r["model"]] = dict(zip(CLASS_LABELS, f1s))
        correct[r["model"]] = [a == b for a, b in zip(y, yh)]

    os.makedirs(OUT, exist_ok=True)
    s = pd.DataFrame(summary)
    pc = pd.DataFrame(per_class).rename(index=CLASS_DISPLAY)
    mc = pd.DataFrame([{"model_a": a, "model_b": b, "a_only_correct": x, "b_only_correct": y, "p_value": p}
                       for a, b in itertools.combinations(correct, 2)
                       for x, y, p in [mcnemar_exact(correct[a], correct[b])]])
    s.to_csv(os.path.join(OUT, f"summary_{args.split}.csv"), index=False)
    pc.to_csv(os.path.join(OUT, f"per_class_f1_{args.split}.csv"))
    mc.to_csv(os.path.join(OUT, f"mcnemar_{args.split}.csv"), index=False)

    md = [f"# Model comparison — `{args.split}` split ({len(common)} documents)\n",
          s.to_markdown(index=False, floatfmt=".3f"), "\n## Per-class F1\n",
          pc.to_markdown(floatfmt=".3f"), "\n## McNemar exact test (pairwise)\n",
          mc.to_markdown(index=False, floatfmt=".3f")]
    with open(os.path.join(OUT, f"comparison_{args.split}.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(s["model"], s["macro_f1"], color="#4C72B0")
        for i, v in enumerate(s["macro_f1"]):
            ax.text(v + 0.005, i, f"{v:.3f}  (real {s['real_correct'][i]})", va="center", fontsize=9)
        ax.set_xlim(0, 1.15); ax.invert_yaxis(); ax.set_xlabel("Macro F1")
        ax.set_title(f"ALIP — {args.split} split ({len(common)} docs)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"macro_f1_{args.split}.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)

    pd.set_option("display.width", 200)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMcNemar:\n" + mc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
