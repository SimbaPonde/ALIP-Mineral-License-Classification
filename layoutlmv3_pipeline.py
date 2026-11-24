"""
ALIP – Tier 2: LayoutLMv3 fine-tuning (rebuilt).

Three stages, each runnable on its own (Colab notebook: notebooks/ALIP_Colab_All_Models.ipynb):

  ocr      Render page 1 of every PDF at 200 DPI and run Tesseract -> words + boxes
           (normalised to 0..1000). Cached to data/ocr_cache/<doc_id>.json + .png
  train    Fine-tune on the `train` split, early-stop on `val` macro-F1, save best
           checkpoint to models/layoutlmv3_best/
  predict  Predict the `test` split (24 synthetic + 10 real) with the best checkpoint
           and write results/predictions/layoutlmv3_fine_tuned__test.json

Fix vs. the original pipeline (report §10.3): words and boxes are passed to the
processor as ONE example (`[words]`, `[boxes]`) with truncation to 512 tokens.
Passing them un-nested makes the processor treat every word as a separate
example (input_ids of shape [n_words, 512]); keeping row 0 then classified each
page from its first word only.

Usage:
    python layoutlmv3_pipeline.py ocr
    python layoutlmv3_pipeline.py train   [--epochs 15 --lr 1e-5 --batch-size 4]
    python layoutlmv3_pipeline.py predict [--split test]
    python layoutlmv3_pipeline.py all
"""

import argparse
import json
import os
import random
import time

import numpy as np

from alip_common import CLASS_LABELS, REPO_ROOT, abspath, load_manifest, save_predictions

OCR_DIR = os.path.join(REPO_ROOT, "data", "ocr_cache")
MODEL_OUT = os.path.join(REPO_ROOT, "models", "layoutlmv3_best")
LABEL2ID = {c: i for i, c in enumerate(CLASS_LABELS)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Stage 1: OCR ─────────────────────────────────────────────────────────────

def ocr_document(pdf_path, dpi=200):
    import pytesseract
    from pdf2image import convert_from_path

    image = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)[0].convert("RGB")
    w, h = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word or float(data["conf"][i]) < 0:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [int(1000 * x / w), int(1000 * y / h), int(1000 * (x + bw) / w), int(1000 * (y + bh) / h)]
        words.append(word)
        boxes.append([min(max(v, 0), 1000) for v in box])
    return image, words, boxes


def stage_ocr(args):
    os.makedirs(OCR_DIR, exist_ok=True)
    df = load_manifest("all")
    for i, row in df.iterrows():
        out = os.path.join(OCR_DIR, row["doc_id"] + ".json")
        if os.path.exists(out) and not args.force:
            continue
        image, words, boxes = ocr_document(abspath(row["file_path"]), dpi=args.dpi)
        image.thumbnail((1000, 1000))          # processor resizes to 224 anyway
        image.save(os.path.join(OCR_DIR, row["doc_id"] + ".png"))
        with open(out, "w") as f:
            json.dump({"doc_id": row["doc_id"], "words": words, "boxes": boxes}, f)
        print(f"[{i + 1}/{len(df)}] {row['doc_id']}: {len(words)} words")


# ── Dataset ──────────────────────────────────────────────────────────────────

def encode(processor, doc_id):
    from PIL import Image
    with open(os.path.join(OCR_DIR, doc_id + ".json")) as f:
        d = json.load(f)
    image = Image.open(os.path.join(OCR_DIR, doc_id + ".png")).convert("RGB")
    words = d["words"] or ["[EMPTY]"]
    boxes = d["boxes"] or [[0, 0, 0, 0]]
    enc = processor(image, [words], boxes=[boxes], truncation=True, padding="max_length",
                    max_length=512, return_tensors="pt")
    assert enc["input_ids"].shape == (1, 512), enc["input_ids"].shape   # guard against §10.3
    return {k: v.squeeze(0) for k, v in enc.items()}


class DocDataset:
    def __init__(self, processor, df):
        self.items = [(encode(processor, d), LABEL2ID[l]) for d, l in zip(df["doc_id"], df["label"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        enc, y = self.items[i]
        return {**enc, "labels": y}


def collate(batch):
    import torch
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "labels"}
    out["labels"] = torch.tensor([b["labels"] for b in batch])
    return out


def run_eval(model, loader, device):
    import torch
    model.eval()
    preds, gold, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            loss_sum += torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum").item()
            n += len(labels)
            preds += logits.argmax(-1).cpu().tolist()
            gold += labels.tolist()
    run_eval.last_loss = loss_sum / max(n, 1)
    return preds, gold


# ── Stage 2: train ───────────────────────────────────────────────────────────

def stage_train(args):
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from transformers import (LayoutLMv3ForSequenceClassification, LayoutLMv3Processor,
                              get_cosine_schedule_with_warmup)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(CLASS_LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)

    train_ds = DocDataset(processor, load_manifest("train"))
    val_ds = DocDataset(processor, load_manifest("val"))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)

    # Model selection: highest val macro-F1, ties broken by lower val loss.
    # (Val F1 saturates at 1.0 on synthetic data within ~2 epochs; selecting on F1
    # alone kept a barely-trained checkpoint.)
    best, best_loss, bad_epochs, history = -1.0, float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for batch in train_dl:
            out = model(**{k: v.to(device) for k, v in batch.items()})
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            loss_sum += out.loss.item()
        preds, gold = run_eval(model, val_dl, device)
        f1 = f1_score(gold, preds, average="macro", labels=list(range(len(CLASS_LABELS))), zero_division=0)
        val_loss = run_eval.last_loss
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_dl), "val_loss": val_loss,
                        "val_macro_f1": f1})
        print(f"epoch {epoch:2d}  loss={loss_sum / len(train_dl):.4f}  val_loss={val_loss:.4f}  "
              f"val_f1={f1:.4f}  ({time.time() - t0:.0f}s)")
        if f1 > best or (f1 == best and val_loss < best_loss - 1e-4):
            best, best_loss, bad_epochs = f1, val_loss, 0
            model.save_pretrained(args.out); processor.save_pretrained(args.out)
            print(f"   saved (epoch {epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (patience={args.patience})")
                break
    with open(os.path.join(args.out, "train_history.json"), "w") as f:
        json.dump({"best_val_macro_f1": best, "best_val_loss": best_loss, "history": history, "args": vars(args)}, f, indent=2)
    print(f"best val macro-F1 = {best:.4f} (val loss {best_loss:.4f}); checkpoint in {args.out}")


# ── Stage 3: predict ─────────────────────────────────────────────────────────

def stage_predict(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LayoutLMv3Processor.from_pretrained(args.out, apply_ocr=False, local_files_only=True)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(args.out, local_files_only=True).to(device)
    df = load_manifest(args.split)
    preds, _ = run_eval(model, DataLoader(DocDataset(processor, df), batch_size=8, collate_fn=collate), device)
    rows = [{"doc_id": d, "true": y, "pred": ID2LABEL[p], "source": s}
            for d, y, p, s in zip(df["doc_id"], df["label"], preds, df["source"])]
    meta = {"checkpoint": os.path.relpath(args.out, REPO_ROOT), "base_model": args.model_name}
    hist = os.path.join(args.out, "train_history.json")
    if os.path.exists(hist):
        meta["training"] = json.load(open(hist))
    path = save_predictions(args.model_label, 2, args.split, rows, meta)
    acc = sum(r["true"] == r["pred"] for r in rows) / len(rows)
    print(f"{args.model_label}: {args.split} accuracy={acc:.3f} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ocr", "train", "predict", "all"])
    ap.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    ap.add_argument("--model-label", default="LayoutLMv3 (fine-tuned)")
    ap.add_argument("--out", default=MODEL_OUT)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--force", action="store_true", help="redo OCR even if cached")
    args = ap.parse_args()

    if args.stage in ("ocr", "all"):
        stage_ocr(args)
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("predict", "all"):
        stage_predict(args)


if __name__ == "__main__":
    main()
