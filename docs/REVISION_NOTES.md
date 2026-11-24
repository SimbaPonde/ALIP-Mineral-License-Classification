# ALIP Final Report — Revision Notes (v1 → v2)

Review of `ALIP_Final_Report_v1_original.pdf` (May 2026) against the code and results in this repository.
Items are ordered by severity. **⚠ VERIFY** = needs the team's result JSONs / scripts to confirm; these were *not* changed to new numbers, only reworded or flagged.

---

## A. Critical — affect the validity of the conclusions

### A1. The three tiers were not evaluated on the same test set
The report repeatedly says "all models were evaluated on the same 170-document test set", but Section 7 describes three different protocols:

| Tier | What was actually done | Problem |
|---|---|---|
| TF-IDF | 5-fold stratified CV over all 170 docs | Out-of-fold predictions — valid, but a different protocol |
| LayoutLMv3 | Trained on 109 + early-stopped on 27, then **re-evaluated on all 170** | 136 of the 170 docs were seen during training/model selection → **train/test leakage** |
| Llama 3.1 | Zero-shot on all 170 | Valid (no training), but few-shot prompt examples came from the same synthetic pool |

Table 4 also defines a 34-document test split (10 real + 24 synthetic) that no reported number appears to use.

**Fix in v2:** new §7.5 "Evaluation protocol" states this honestly; "same test set" claims removed. **Recommended re-run:** report all five models on the 34-doc held-out test split *and* on the 10 real documents alone. That is the defensible comparison.

### A2. Label leakage in the synthetic generator
§5.2 claims class-identifying keywords were "deliberately excluded". They were not — every generator writes the class name as the document title (`generate_synthetic_docs.py`, lines 257, 366, 458, 554, 639, 761, 872, 993: `"EXPLORATION LICENSE"`, `"PROSPECTING PERMIT"`, `"MINING LEASE"`, `"APPLICATION FOR RENEWAL OF …"`, etc.). §6.1 even says TF-IDF learns "hereby granted exploration rights". This explains the perfect 1.000 scores on synthetic classes.

**Fix in v2:** claim removed; leakage stated as a limitation. **Recommended:** add a `--strip-titles` mode to the generator and re-run as an ablation.

### A3. "Llama outperforms TF-IDF" is not supported
0.989 vs 0.984 = **168 vs 167 correct documents — a one-document difference**, with no significance test. "Surprising and publishable finding" and "outperforms both" overclaim.

**Fix in v2:** reworded to "TF-IDF and zero-shot Llama are statistically indistinguishable on this data; both beat LayoutLMv3". McNemar's test on the 10 real docs is trivially cheap — recommend running it rather than deferring.

### A4. Dataset counts don't add up
- Abstract: "170 documents combining 160 synthetic … with 27 real" → 160 + 27 = 187. The 170 is 160 synthetic + **10** real Tier 1.
- §5.3: 27 real = 10 Tier 1 + 14 Tier 2 = 24 → **3 documents unaccounted for**. ⚠ VERIFY.
- §4 / Table 9: Tier 2 docs were said to form an "other" catch-all class in the test set, but all models are 8-class and the 170 total excludes them. **Fix in v2:** stated that the 14 Tier 2 docs were excluded from the 8-class evaluation.

---

## B. Numerical inconsistencies (derived from per-class F1 + class support)

Supports: NLA 27 (20 syn + 7 real), SSMP 22, Renewal 21, others 20.

### B1. TF-IDF errors are described backwards
Text: "three misclassifications all occurred in New License Application". The per-class F1s only reconcile one way: SSMP 0.952 → TP 20, FN 2, FP 0; Renewal 0.976 → TP 20, FN 1; NLA 0.947 → TP 27, FP 3. So **2 real SSMP + 1 real Renewal doc were predicted as NLA** — the NLA class absorbed errors; no NLA doc was misclassified. **Fixed in v2.**

### B2. LayoutLMv3 Exploration License
Text says Exploration License (F1 = 0.833) errors occur "where real test documents differ most from synthetic" — but Exploration License has **zero real documents**. Also, 3 errors on a 20-doc class can't give F1 = 0.833 (minimum with 3 errors ≈ 0.919). ⚠ VERIFY against `layoutlmv3_reeval` JSON. Sentence removed in v2.

### B3. Llama zero-shot per-class table
With 2 errors total, both NLA (NLA F1 0.962 ✓), the only way Renewal is < 1.000 is if both NLA docs were predicted as **Renewal Application** → Renewal F1 = 0.955 (table says 0.952), macro recall 0.9907 ✓. So the errors were NLA → Renewal, not "the specific license type being applied for". ⚠ VERIFY; v2 text changed to match the arithmetic and flagged.

### B4. TF-IDF config mismatch
Report: max 10,000 features. Code: `max_features=5000`, `min_df=2`, `sublinear_tf=True`. **Fixed in v2** (code is source of truth).

### B5. Repo results ≠ report results
`results/baseline_results.csv` shows F1 = 1.000 on 160 synthetic docs — the synthetic-only run, not the 170-doc unified run. The unified dataset, merger script, LayoutLMv3 scripts, `llm_evaluation.py`, and `compare_models.py` are referenced in the report but **not in this upload**. README notes this.

---

## C. Technical corrections

| § | Original | Correction |
|---|---|---|
| 6.1 | LR "passes the result through a sigmoid" | Multiclass LR (lbfgs) uses **softmax** |
| 6.1 | "SVMs handle class imbalance slightly better than LR" | Unsupported; removed |
| 6.2 / Table 5 | LayoutLMv3 125M params | LayoutLMv3-**base = 133M** (Huang et al., 2022) |
| 6.3 | Llama trained on "legal documents, government publications" | Meta doesn't publish a breakdown; reworded |
| 1 | Llama as "upper-bound reference" | It's a strong reference/challenger, not an upper bound; reworded |
| 5.2 | `generate_synthetic_docs_v3.py` | File is `generate_synthetic_docs.py` |
| 10.2 | `rel.replace('\\\\', '/')` | That literal is *two* backslashes; should be `rel.replace('\\', '/')` |
| 10.3 | Overflow "fix": take `enc[k][0:1]` | **Likely a bug, not a fix.** `[383, 512]` ≈ one row per *word* — the classic symptom of passing `words`/`boxes` un-nested, so the processor treats each word as a separate example. Taking row 0 would classify each page from its **first word only**, which could fully explain LayoutLMv3's lower score. Correct call: `processor(image, [words], boxes=[boxes], truncation=True, max_length=512, return_tensors="pt")`. ⚠ VERIFY and re-run |
| 6.2 | Image "patches" via `LayoutLMv3ImageProcessor` overflow | Overflow comes from the tokenizer, not the image processor |
| Few-shot | Explained as synthetic examples mismatching real docs | 160/170 test docs are synthetic, and the gap is 2 docs; explanation is speculative — reworded |

## D. Presentation

- Table numbering: text references "Table 7" (main results) and "Table 8" (per-class) but they are captioned 6 and 7; §5.1 table uncaptioned. Renumbered in v2.
- Title box reads "KEY FINDING / KEY RESULT" (duplicated label).
- No in-text citations — references [1]–[15] are never cited. v2 adds citations.
- Ref [2] uses the **Llama 2** paper title. Correct: Grattafiori/Dubey et al. (2024), *The Llama 3 Herd of Models*, arXiv:2407.21783.
- Ref [4] (Naive Bayes) and [15] (EU AI Act) are unrelated to anything in the text; removed.
- Ref [3]: NIPS 2017 → NeurIPS naming fine, kept.
- Table 8: Anthropic consumer policy — consumer chats are used for training only if the user allows it (opt-in/opt-out choice introduced 2025); reworded.
- Goals table claims "synthetic ceiling vs real-doc drop documented" — no real-only metrics are reported. Add a real-docs-only accuracy row (10 docs) per model.

## E. Recommended re-runs before this goes in a portfolio/paper
1. Fix the LayoutLMv3 processor call (C, §10.3) and retrain.
2. Evaluate all models on the 34-doc held-out split and on the 10 real docs alone.
3. Regenerate synthetic docs without title text; re-run TF-IDF as an ablation.
4. McNemar's test (TF-IDF vs Llama vs LayoutLMv3) on held-out predictions.
5. Commit the missing scripts (data merger, LayoutLMv3 prep/finetune/reeval, LLM eval, compare).

## F. References changed in v2
- Llama 3 → *The Llama 3 Herd of Models* (arXiv:2407.21783).
- Tesseract → Smith (2007), ICDAR (the engine paper) instead of the multilingual-adaptation paper.
- Botswana: Cap. 66:01 is the **Mines and Minerals Act**, not the "Mines, Quarries, Works and Machinery Act" (that is Cap. 44:02). ⚠ VERIFY which act the documents were issued under.
- LayoutLMv2: published at ACL-IJCNLP 2021 (was cited as arXiv 2020).
- Added LIBLINEAR (LinearSVC backend). Removed uncited Naive Bayes and EU AI Act refs.
- All references now cited in the text.

---

## G. Rebuild log (Nov 2026)

The original Tier 2/3 scripts, the dataset merger and their result JSONs were lost. The pipeline was rebuilt from the report's description, using the 27 real forms that survived:

- `data/real_labels.csv`: labels reconstructed from the form titles. They reproduce the report's counts exactly (7 NLA, 2 SSMP, 1 Renewal, 14 Tier 2, 3 out of scope: processing/smelting/refining licences). This also resolves item A4: the "3 unaccounted" documents are Uganda Forms 36, 39 and 41.
- `build_unified_dataset.py`: fixed split (109 / 27 / 34), real docs in test only, forward-slash paths (fixes §10.2 at the source).
- `layoutlmv3_pipeline.py`: processor called with `[words]`, `[boxes]` plus an assertion that inputs are `[1, 512]` (fixes the §10.3 bug). Smoke-tested end-to-end on CPU with a tiny random model; the real `microsoft/layoutlmv3-base` run needs Colab.
- `llm_evaluation.py`: Ollama `/api/chat`, temperature 0, JSON-schema-constrained output with the original parse cascade as fallback, few-shot examples drawn from `train` only, per-document resume cache.
- `compare_models.py`: scores every model on identical documents, reports real-doc accuracy separately, plus exact McNemar tests.

**First held-out result (TF-IDF):** 31/34 test docs, macro F1 0.929; **7/10 real**. The errors are Uganda Form 19 and Form 21 (SSMP → NLA) and Form 25 (Renewal → SSMP), the same three documents implied by the original per-class table (B1). Masking class names changes nothing. Note: NLTK's lemmatizer data could not be downloaded in the rebuild environment, so the script's suffix-stripping fallback was used; rerun locally for the exact configuration.

The headline numbers in the v2 report remain the original May 2026 figures. Once the LayoutLMv3 and Llama held-out runs are done, the report's §8 should be replaced with `results/comparison/comparison_test.md`.

## H. v3 report (Nov 2026)

All held-out runs are complete, and the report has been updated to v3 (§7–8 rewritten, abstract, conclusion, goals, risks and future work updated; the May 2026 numbers are kept as §8.4 for reference). This resolves every VERIFY item: the original per-class figures are superseded rather than reconciled.

| Model | Macro F1 | Real correct |
|---|---|---|
| Llama 3.1 8B few-shot | 0.957 | 8/10 |
| TF-IDF + LR / SVC | 0.929 | 7/10 |
| Llama 3.1 8B zero-shot | 0.919 | 8/10 |
| LayoutLMv3 fine-tuned | 0.769 | 1/10 |

LayoutLMv3 run log: OCR, 15 epochs, best checkpoint = epoch 15 (val loss 0.030; val F1 1.000 from epoch 2). The checkpoint-selection fix (ties broken by val loss) did not change the outcome: the epoch-2 checkpoint scored 26/34, the epoch-15 checkpoint 25/34.
