"""
ALIP – Tier 3: Llama 3.1 8B zero-shot / few-shot classification via a local Ollama server (rebuilt).

No document text leaves the machine: requests go only to the local Ollama HTTP API.

Setup (once):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3.1:8b
    3. Ollama runs in the background on http://localhost:11434

Usage:
    python llm_evaluation.py --mode zero-shot
    python llm_evaluation.py --mode few-shot
    python llm_evaluation.py --mode zero-shot --split all      # all 170 docs
    python llm_evaluation.py --mode zero-shot --limit 3        # quick check

Progress is cached per document in results/llm_cache/, so an interrupted run
resumes where it stopped (CPU inference is ~1–2 min per document).
"""

import argparse
import json
import os
import re
import time

import requests

from alip_common import (CLASS_DESCRIPTIONS, CLASS_DISPLAY, CLASS_LABELS, REPO_ROOT, abspath,
                         extract_text, load_manifest, save_predictions)

CACHE_DIR = os.path.join(REPO_ROOT, "results", "llm_cache")
MAX_CHARS = 2500
FEWSHOT_CHARS = 700

SYSTEM = ("You are an expert in African mineral licensing regulation. You classify documents "
          "submitted to a mining cadastre office into exactly one category.")


def class_block():
    return "\n".join(f"- {c}: {CLASS_DISPLAY[c]} — {CLASS_DESCRIPTIONS[c]}" for c in CLASS_LABELS)


def fewshot_examples(seed=42):
    """One synthetic training document per class (train split only — never test docs)."""
    df = load_manifest("train")
    out = []
    for c in CLASS_LABELS:
        row = df[df["label"] == c].sample(1, random_state=seed).iloc[0]
        out.append((c, extract_text(abspath(row["file_path"]))[:FEWSHOT_CHARS]))
    return out


def build_prompt(text, examples=None):
    p = [f"Categories:\n{class_block()}\n"]
    if examples:
        p.append("Examples:")
        for label, ex in examples:
            p.append(f"--- DOCUMENT (excerpt) ---\n{ex}\n--- CATEGORY: {label}\n")
    p.append(f"Classify this document.\n--- DOCUMENT ---\n{text[:MAX_CHARS]}\n--- END ---\n")
    p.append('Answer with JSON only: {"category": "<one of the category ids above>"}')
    return "\n".join(p)


def parse_label(response: str):
    """JSON first, then exact id, then display name, then keyword cascade."""
    r = response.strip()
    try:
        cat = json.loads(re.search(r"\{.*\}", r, re.S).group(0)).get("category", "")
        if cat in CLASS_LABELS:
            return cat, "json"
        r = str(cat) or r
    except Exception:
        pass
    low = r.lower()
    for c in CLASS_LABELS:
        if c in low:
            return c, "id"
    for c in CLASS_LABELS:
        if CLASS_DISPLAY[c].lower() in low:
            return c, "display_name"
    keywords = [("renew", "renewal_application"), ("compliance", "environmental_compliance_certificate"),
                ("impact", "environmental_impact_assessment"), ("small", "small_scale_mining_permit"),
                ("artisanal", "small_scale_mining_permit"), ("application", "new_license_application"),
                ("exploration", "exploration_license"), ("prospect", "prospecting_permit"),
                ("lease", "mining_lease"), ("mining", "mining_lease")]
    for k, c in keywords:
        if k in low:
            return c, "keyword"
    return "unparsed", "failed"


def call_ollama(host, model, prompt, timeout):
    body = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 8192},
        "format": {"type": "object", "properties": {"category": {"type": "string", "enum": CLASS_LABELS}},
                   "required": ["category"]},
    }
    r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    df = load_manifest(args.split)
    if args.limit:
        df = df.head(args.limit)
    examples = fewshot_examples() if args.mode == "few-shot" else None
    cache_dir = os.path.join(CACHE_DIR, f"{args.model.replace(':', '_')}_{args.mode}")
    os.makedirs(cache_dir, exist_ok=True)

    rows, times = [], []
    for i, row in df.iterrows():
        cache = os.path.join(cache_dir, row["doc_id"] + ".json")
        if os.path.exists(cache):
            rec = json.load(open(cache))
        else:
            text = extract_text(abspath(row["file_path"]))
            t0 = time.time()
            raw = call_ollama(args.host, args.model, build_prompt(text, examples), args.timeout)
            pred, how = parse_label(raw)
            rec = {"doc_id": row["doc_id"], "raw": raw, "pred": pred, "parse": how,
                   "seconds": round(time.time() - t0, 1)}
            json.dump(rec, open(cache, "w"), indent=1)
        times.append(rec["seconds"])
        ok = "✓" if rec["pred"] == row["label"] else "✗"
        print(f"[{i + 1}/{len(df)}] {ok} {row['doc_id']:40s} {row['label']:38s} -> {rec['pred']} ({rec['seconds']}s)")
        rows.append({"doc_id": row["doc_id"], "true": row["label"], "pred": rec["pred"],
                     "source": row["source"], "parse": rec["parse"]})

    name = f"Llama 3.1 8B ({args.mode})" if args.model == "llama3.1:8b" else f"{args.model} ({args.mode})"
    meta = {"ollama_model": args.model, "mode": args.mode, "max_chars": MAX_CHARS,
            "mean_seconds_per_doc": round(sum(times) / max(len(times), 1), 1)}
    if args.limit:
        print("--limit set: not writing a predictions file.")
        return
    path = save_predictions(name, 3, args.split, rows, meta)
    acc = sum(r["true"] == r["pred"] for r in rows) / len(rows)
    print(f"{name}: {args.split} accuracy={acc:.3f} -> {path}")


if __name__ == "__main__":
    main()
