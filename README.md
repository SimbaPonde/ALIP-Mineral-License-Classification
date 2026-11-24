# ALIP – Aquila Land Intelligence Platform

**Mineral license document classification: a three-tier experimental study.**
Capstone project, M.S. Computer Science, Katz School of Science and Health, Yeshiva University (2026).
Team: Enock Katenda · Ezekiel (Simba) Ponde · Tafadzwa Nyemba

ALIP is an intelligent intake layer for mining cadastre systems (e.g. Trimble Landfolio, MCAS). It classifies incoming mineral licensing documents (exploration licenses, mining leases, applications, EIAs, …) so they don't have to be read and keyed in by hand. Three approaches are compared on 8 document classes:

| Tier | Model | Role |
|---|---|---|
| 1 | TF-IDF + Logistic Regression / LinearSVC | Classical baseline |
| 2 | LayoutLMv3-base, fine-tuned (text + layout + image) | Domain fine-tuning (research hypothesis) |
| 3 | Llama 3.1 8B zero-/few-shot, run locally via Ollama | General-purpose LLM, no training data |

## Results

### Original submission (May 2026; superseded, protocols differed per tier)

| Model | Macro F1 | Correct / 170 |
|---|---|---|
| TF-IDF + LR / LinearSVC (5-fold CV) | 0.984 | 167 |
| LayoutLMv3 fine-tuned (evaluated on docs incl. its training data) | 0.943 | 160 |
| Llama 3.1 8B zero-shot | 0.989 | 168 |
| Llama 3.1 8B few-shot | 0.979 | 166 |

See [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md) for why these aren't directly comparable.

### Held-out evaluation (Nov 2026) — current results

Every model trains only on the 109 synthetic `train` docs, and is scored on the **same 34-doc `test` split** (24 synthetic + all 10 real government forms). Full table: [`results/comparison/comparison_test.md`](results/comparison/comparison_test.md).

| Model | Macro F1 | Accuracy | Real docs correct | Status |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.929 | 0.912 | 7/10 | ✅ done |
| TF-IDF + LinearSVC | 0.929 | 0.912 | 7/10 | ✅ done |
| TF-IDF + LR, class names masked (leakage ablation) | 0.929 | 0.912 | 7/10 | ✅ done |
| LayoutLMv3 (fine-tuned, processor bug fixed) | 0.769 | 0.735 | **1/10** | ✅ done |
| Llama 3.1 8B zero-shot | 0.919 | 0.882 | 8/10 | ✅ done |
| Llama 3.1 8B few-shot | 0.957 | 0.941 | 8/10 | ✅ done |

**Few-shot Llama is best; fine-tuned LayoutLMv3 is worst** and significantly worse than few-shot Llama (exact McNemar p = 0.039). LayoutLMv3 gets all 24 synthetic docs right but only 1/10 real forms: with ~14 template-generated training docs per class it memorises template layouts. TF-IDF and Llama can't be separated statistically (p ≥ 0.69). Zero-shot Llama labels *applications for* a prospecting licence as a prospecting permit (4 errors: 2 synthetic, 2 real); one example per class in the prompt (few-shot) fixes this. Every TF-IDF error is on a real form (2 small-scale/artisanal licence applications → New License Application; the renewal form → Small-Scale Mining Permit). With only 10 real documents, differences of a few documents between models will not be statistically significant — `compare_models.py` reports exact McNemar tests.

## Reproduce

**Easiest:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SimbaPonde/ALIP-Mineral-License-Classification/blob/main/notebooks/ALIP_Colab_All_Models.ipynb) → T4 GPU → Run all. Runs every tier (TF-IDF, LayoutLMv3, Llama via Ollama) and downloads the results.

Locally:

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python generate_synthetic_docs.py                       # 160 synthetic PDFs -> output/ (already committed)
python build_unified_dataset.py                         # 170-doc manifest + fixed train/val/test split

python baseline_heldout.py                              # Tier 1
python baseline_heldout.py --mask-class-names           # Tier 1 leakage ablation
python llm_evaluation.py --mode zero-shot               # Tier 3 (needs Ollama + `ollama pull llama3.1:8b`)
python llm_evaluation.py --mode few-shot
# Tier 2:
python layoutlmv3_pipeline.py all                       # needs tesseract-ocr + poppler, ideally a GPU

python compare_models.py                                # tables, per-class F1, McNemar, plot
```

Each model writes `results/predictions/<model>__test.json`; `compare_models.py` scores whatever files are present on the documents they share. The Llama script caches every response in `results/llm_cache/`, so an interrupted CPU run resumes where it stopped.

## Repository layout

```
alip_common.py               # class list, descriptions, shared I/O and prediction format
generate_synthetic_docs.py   # synthetic PDF generator (8 classes × 20, 4 jurisdictions)
build_unified_dataset.py     # merges synthetic + real Tier 1 docs, fixed split -> data/unified_manifest.csv
baseline_heldout.py          # Tier 1 on held-out split (+ class-name masking ablation)
baseline_pipeline.py         # Tier 1 5-fold CV analysis (original)
layoutlmv3_pipeline.py       # Tier 2: OCR -> fine-tune -> predict
llm_evaluation.py            # Tier 3: Llama 3.1 via local Ollama, zero-/few-shot
compare_models.py            # identical-document comparison, McNemar tests
notebooks/                   # Colab notebook that runs all tiers
data/real/                   # 27 public government forms (Botswana ×4, Uganda ×23)
data/real_labels.csv         # labels: 10 Tier 1, 14 Tier 2, 3 outside the taxonomy
output/                      # 160 synthetic PDFs + manifest
results/                     # predictions, comparison tables, original synthetic-only CV
docs/                        # final report (v1 original, v3 current) + revision notes
```

## Data notes

- **Real documents** are blank statutory forms published by Botswana's Ministry of Minerals and Uganda's Ministry of Energy and Mineral Development (Mining and Minerals (Licensing) Regulations, 2023). They contain no personal or company data. Labels in `data/real_labels.csv` were reconstructed in Nov 2026 and reproduce the original split (7 New License Application, 2 Small-Scale Mining Permit, 1 Renewal; 14 Tier 2; 3 out of scope). Form VII (Botswana minerals permit) is the one ambiguous case.
- **All 10 real Tier 1 documents are application forms.** No real issued licence, lease, EIA or compliance certificate is in the dataset, so five classes are evaluated on synthetic data only.
- **Synthetic documents print their class name as the title** (label leakage). The `--mask-class-names` ablation removes class-naming phrases; synthetic accuracy stays at 100%, so the templates are separable even without the titles.

## Documents

- [`docs/ALIP_Final_Report_v3.pdf`](docs/ALIP_Final_Report_v3.pdf): final report, revised with held-out results ([Markdown](docs/ALIP_Final_Report_v3.md))
- [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md): review of v1, rebuild log, v3 changes
- [`docs/ALIP_Final_Report_v1_original.pdf`](docs/ALIP_Final_Report_v1_original.pdf): original submission

## Stack
Python · scikit-learn · NLTK · pdfplumber · ReportLab · PyTorch · Hugging Face Transformers (LayoutLMv3) · Tesseract OCR · Ollama (Llama 3.1 8B) · Google Colab
