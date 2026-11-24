# ALIP – Aquila Land Intelligence Platform

**Mineral license document classification: a three-tier experimental study.**
Capstone project, M.S. Computer Science, Katz School of Science and Health, Yeshiva University (2026).
Team: Enock Katenda · Ezekiel (Simba) Ponde · Tafadzwa Nyemba

ALIP is an intelligent intake layer for mining cadastre systems (e.g. Trimble Landfolio, MCAS). It automatically classifies incoming mineral licensing documents (exploration licenses, mining leases, applications, EIAs, etc.) so they don't have to be read and keyed in by hand. We compare three approaches on 8 document classes:

| Tier | Model | Macro F1 | Correct / 170 |
|---|---|---|---|
| 1 | TF-IDF + Logistic Regression / LinearSVC | 0.984 | 167 |
| 2 | LayoutLMv3-base, fine-tuned | 0.943 | 160 |
| 3 | Llama 3.1 8B zero-shot (local, Ollama) | 0.989 | 168 |
| 3 | Llama 3.1 8B few-shot | 0.979 | 166 |

**Finding:** fine-tuning LayoutLMv3 did **not** beat the lexical baseline. A locally hosted zero-shot LLM matched the baseline with no labelled data and no external data transfer. See the report for limitations: the tiers used different evaluation protocols, the synthetic data has title-level label leakage, and only 3 of 8 classes have real test documents.

## Documents

- [`docs/ALIP_Final_Report_v2.pdf`](docs/ALIP_Final_Report_v2.pdf): revised final report ([Markdown source](docs/ALIP_Final_Report_v2.md))
- [`docs/REVISION_NOTES.md`](docs/REVISION_NOTES.md): what changed from v1 and why, plus the recommended re-runs
- [`docs/ALIP_Final_Report_v1_original.pdf`](docs/ALIP_Final_Report_v1_original.pdf): the original submission

## Repository contents

```
generate_synthetic_docs.py   # Tier 1 synthetic PDF generator (8 classes × 20, 4 jurisdictions)
baseline_pipeline.py         # TF-IDF + LR / LinearSVC, 5-fold stratified CV, plots
output/                      # 160 generated synthetic PDFs + manifest.csv
results/                     # Baseline results on the 160 synthetic docs (F1 = 1.000, see note)
docs/                        # Reports and revision notes
```

> **Note:** `results/` is from the synthetic-only baseline run (160 docs), where the class title printed on every synthetic document makes the task trivial. The 170-doc unified results in the report also include the 10 real government documents. Those documents, the dataset-merge script, the LayoutLMv3 scripts (`layoutlmv3_data_prep.py`, `layoutlmv3_finetune.py`, `layoutlmv3_reeval.py`), `llm_evaluation.py` and `compare_models.py` are not yet committed.

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python generate_synthetic_docs.py                 # writes ./output
python baseline_pipeline.py --data ./output --results ./results
```

## Stack
Python · scikit-learn · NLTK · pdfplumber · ReportLab · Hugging Face Transformers (LayoutLMv3) · Tesseract OCR · Ollama (Llama 3.1 8B) · Google Colab (T4)
