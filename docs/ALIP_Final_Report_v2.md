---
title: "ALIP – Aquila Land Intelligence Platform"
subtitle: "Mineral License Document Classification: A Three-Tier Experimental Study (Revised v2)"
author:
  - "Enock Katenda · Ezekiel Ponde · Tafadzwa Nyemba"
  - "Katz School of Science and Health, Yeshiva University, New York, NY"
date: "Capstone Project — May 2026 (revised November 2026)"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

> **Key result.** On 170 mineral licensing documents (160 synthetic, 10 real), zero-shot Llama 3.1 8B (macro F1 = 0.989) and a TF-IDF baseline (0.984) perform within one document of each other, and both outperform fine-tuned LayoutLMv3 (0.943). The fine-tuning hypothesis was not supported. Because the three tiers used different evaluation protocols and the synthetic data contains title-level label leakage (§7.5, §10), these figures are an optimistic upper estimate, not a measure of real-world generalisation.

## Abstract

ALIP investigates whether domain-specific fine-tuning measurably improves Intelligent Document Processing (IDP) for mineral licensing documents in a low-resource African regulatory context. We construct a dataset of 170 documents: 160 synthetic PDFs across eight Tier 1 classes and 10 real Tier 1 government forms from Botswana and Uganda (drawn from 27 real documents collected). Three model tiers are evaluated: (1) TF-IDF with Logistic Regression / LinearSVC; (2) fine-tuned LayoutLMv3-base [1]; and (3) Llama 3.1 8B [2] in zero-shot and few-shot modes, run locally via Ollama [8]. Contrary to the central hypothesis, fine-tuned LayoutLMv3 scored lowest (macro F1 = 0.943). Zero-shot Llama (0.989) and the TF-IDF baseline (0.984) were effectively tied — a one-document difference. Few-shot prompting (0.979) did not improve on zero-shot. We document important limitations — evaluation protocols that differ across tiers, train/test overlap in the LayoutLMv3 re-evaluation, class titles present in synthetic documents, and only three of eight classes having any real test documents — and specify the re-runs needed to turn these results into a generalisation claim. The practical takeaway is that, for descriptively distinct regulatory document classes, a cheap lexical baseline or a locally hosted open-weight LLM is a strong starting point before investing in layout-aware fine-tuning.

# 1. Project Overview

ALIP investigates whether domain-specific fine-tuning can measurably improve IDP performance in the mineral licensing domain — a low-resource setting with scarce labelled data, varied formats across jurisdictions, and limited public availability of government records.

**Research question:** Can a domain-specifically fine-tuned model significantly outperform a general-purpose baseline on document classification within the mineral licensing domain?

The study uses three tiers: a classical TF-IDF baseline as the performance floor, fine-tuned LayoutLMv3 as the research contribution, and Llama 3.1 8B as a zero-shot general-purpose reference that tests whether fine-tuning is needed at all.

> **Scope.** The project targets mineral licensing documents only and is not intended to generalise across domains.

# 2. Existing Systems and the Research Gap

## 2.1 What already exists
Trimble Landfolio (formerly FlexiCadastre) [9] is the dominant mining cadastre platform, deployed in Zambia, Angola, Botswana, Namibia, Ethiopia and Uganda, managing the licence lifecycle with configurable workflows, ArcGIS mapping and public e-Government portals. The open-source Mining Cadastre Administration System (MCAS) [10] provides comparable lifecycle management with GeoServer GIS. In the United States, the Bureau of Land Management operates MLRS [11]. Industry tools such as Pandell LandWorks and mineral.tech serve companies with portfolio tracking, royalty auditing and reporting.

## 2.2 The gap
These systems orchestrate workflows and store records; none automatically reads or classifies incoming documents. A human operator must read each document, determine its type and key in data. Consequences include intake backlogs, classification errors in understaffed ministries, and scanned legacy documents that are stored but never understood.

## 2.3 What ALIP addresses
ALIP is an intelligent intake layer upstream of platforms such as Landfolio or MCAS: it automates the first step — deciding what a document is.

**Table 1: Capability comparison — existing systems vs. ALIP.**

| Capability | Existing systems (Landfolio / MCAS) | ALIP |
|---|---|---|
| Automatic document classification | Not available | Core contribution |
| Domain-specific NLP / ML understanding | Not available | Core contribution |
| Low-resource fine-tuning research | Not available | Core contribution |
| Automated intake triage | Not available | Target capability |

# 3. Project Goals and Achievements

The primary goal was to design and evaluate a domain-specific fine-tuning pipeline benchmarked against a classical baseline and a general-purpose LLM. All three tiers were implemented and evaluated.

**Table 2: Secondary goals — planned vs. achieved.**

| Goal | Status | Outcome |
|---|---|---|
| Labelled dataset covering ≥ 6 Tier 1 classes | Achieved | 8 classes; 170 docs (160 synthetic + 10 real) |
| Reproducible TF-IDF + LR/SVM baseline | Achieved | Macro F1 = 0.984 (5-fold CV) |
| Fine-tune LayoutLMv3 and compare | Achieved, with caveats | Macro F1 = 0.943; evaluation overlaps training data (§7.5) |
| Accuracy, precision, recall, macro-F1 | Achieved | Reported for all five models |
| Document impact of synthetic data | Partial | Synthetic-only F1 = 1.000; real-only metrics not yet reported separately |
| General-purpose LLM reference | Achieved | Llama 3.1 8B zero-shot F1 = 0.989 |

# 4. Document Taxonomy

We compiled a 44-class taxonomy of mineral licensing documents from African, Australian and Canadian regulatory frameworks (Table 3). The eight Tier 1 classes were used in the experiments; Tier 2 classes were out of scope for the 8-class models.

**Table 3: Complete 44-class document taxonomy.**

| Group | Document class | Description | Tier |
|---|---|---|---|
| A. Licensing & permit instruments | Exploration License | Authorises geological/geophysical survey over a defined area | **1** |
| | Prospecting Permit | Permits initial investigation for presence of minerals | **1** |
| | Mining Lease / License | Grants rights to extract minerals at commercial scale | **1** |
| | Small-Scale Mining Permit | Artisanal or small-scale mining operations | **1** |
| | Special Mining Lease | Large strategic deposits; negotiated with government | 2 |
| | Retention License | Retains rights over a deposit not yet commercially viable | 2 |
| | Exclusive Prospecting Order | Exclusive prospecting rights over a large area | 2 |
| | Reconnaissance License | Broad, non-exclusive preliminary survey | 2 |
| | Mineral Claim Certificate | Registers a staked claim over a land parcel | 2 |
| B. Application & renewal | New License Application | Initial application for any licensing instrument | **1** |
| | Renewal Application | Extends an existing licence before expiry | **1** |
| | Variation Application | Change to area, conditions or duration | 2 |
| | Transfer / Assignment Application | Transfers licence ownership | 2 |
| | Cession Application | Partial transfer or relinquishment of an area | 2 |
| | Amalgamation Application | Merges adjacent licences | 2 |
| | Suspension Application | Temporary suspension of obligations | 2 |
| | Cancellation / Surrender Notice | Voluntary relinquishment | 2 |
| C. Compliance & regulatory | Annual / Work Programme Report | Activities and expenditure in the period | 2 |
| | Environmental Impact Assessment | Pre-mining environmental study | **1** |
| | Environmental Management Plan | Operational mitigation plan | 2 |
| | Environmental Compliance Certificate | Certifies EIA/EMP requirements met | **1** |
| | Rehabilitation Plan | Post-mining land restoration | 2 |
| | Mine Closure Plan | Decommissioning and closure | 2 |
| | Health & Safety Management Plan | Occupational health and safety protocols | 2 |
| | Social Impact Assessment | Effects on local communities | 2 |
| D. Financial & fiscal | Royalty Return / Statement | Production and royalties owed | 2 |
| | License Fee Payment Receipt | Proof of fee payment | 2 |
| | Performance Bond / Guarantee | Security against rehabilitation obligations | 2 |
| | Production Report | Tonnage and grade data | 2 |
| E. Legal & ownership | Deed of Transfer | Effects a licence ownership transfer | 2 |
| | Mortgage / Charge over Mining Rights | Security interest against a licence | 2 |
| | Court Order / Arbitration Award | Decisions affecting licence rights | 2 |
| | Caveat Notice | Blocks dealings pending dispute | 2 |
| | Objection / Opposition Notice | Third-party objection to an application | 2 |
| | Ministerial Consent | Approval for strategic transactions | 2 |
| F. Correspondence & admin | Acknowledgement Letter | Confirms receipt of a submission | 2 |
| | Request for Additional Information | Regulator query | 2 |
| | Notice of Deficiency | Submission incomplete | 2 |
| | Show Cause Notice | Explain non-compliance before penalties | 2 |
| | Approval / Grant Letter | Official grant of a licence or permit | 2 |

# 5. Dataset Construction

## 5.1 Tier 1 class set — planned vs. achieved

**Table 4: Tier 1 class counts.**

| # | Class | Min. target | Synthetic | Real |
|---|---|---|---|---|
| 1 | Exploration License | 20 | 20 | 0 |
| 2 | Prospecting Permit | 20 | 20 | 0 |
| 3 | Mining Lease / License | 20 | 20 | 0 |
| 4 | Small-Scale Mining Permit | 15 | 20 | 2 |
| 5 | New License Application | 20 | 20 | 7 |
| 6 | Renewal Application | 15 | 20 | 1 |
| 7 | Environmental Impact Assessment | 15 | 20 | 0 |
| 8 | Environmental Compliance Certificate | 15 | 20 | 0 |
| | **Total** | **140** | **160** | **10** |

## 5.2 Synthetic document generation
A template-based generator (`generate_synthetic_docs.py`, ReportLab) produces legally formatted PDFs for each class across four jurisdictions (Zimbabwe, Zambia, Botswana, Namibia), with randomised applicants, coordinates, dates, reference numbers and commodities.

**Limitation — label leakage.** Each template prints its class name as the document title (e.g. "EXPLORATION LICENSE", "APPLICATION FOR RENEWAL OF …"), and body text contains class-specific phrases ("hereby grants an Exploration License"). A classifier can therefore succeed on synthetic documents by keyword lookup, which explains the perfect synthetic-class scores in §8. A title-stripped ablation is listed as future work.

## 5.3 Real document collection
27 real government mineral licensing documents were collected from public regulatory portals: 4 from Botswana's Ministry of Mineral Resources [13] and 23 from Uganda's licensing portal under the Mining and Minerals Act 2022 [12]. Of these, 10 map to Tier 1 classes and were used in evaluation; 14 map to Tier 2 classes and were excluded from the 8-class experiments; the remaining 3 were not used. *(VERIFY: the status of these 3.)*

> **Evaluation note.** Five of the eight classes — Exploration License, Prospecting Permit, Mining Lease, EIA and Environmental Compliance Certificate — have no real documents. Scores on those classes validate the pipeline on synthetic data only. The three classes with real documents (New License Application, Small-Scale Mining Permit, Renewal Application) carry the substantive evaluation.

**Table 5: Dataset splits (used for LayoutLMv3 training).**

| Split | Total | Real | Synthetic | Notes |
|---|---|---|---|---|
| Train | 109 | 0 | 109 | Synthetic only |
| Validation | 27 | 0 | 27 | Synthetic only — early stopping |
| Test | 34 | 10 | 24 | 10 real Tier 1 + 24 synthetic |
| **Total** | **170** | **10** | **160** | |

# 6. Methodology and Models

## 6.1 Tier 1 — Classical baseline (TF-IDF + LR / SVM)
*Question: what is the performance floor for lightweight, interpretable models?*

TF-IDF represents each document as a sparse vector in which a term's weight rises with its frequency in the document and falls with the number of documents containing it, so frequent-but-ubiquitous words carry little weight. **Logistic Regression** learns a weight per term per class and converts the class scores to probabilities with a softmax. It is well suited to settings where features greatly outnumber training examples, as here. **LinearSVC** fits maximum-margin linear boundaries, one-vs-rest across the eight classes [14]. Both are implemented in scikit-learn [7].

*Implementation.* Text is extracted with pdfplumber and preprocessed with NLTK (lower-casing, stop-word removal, WordNet lemmatisation). The vectoriser uses unigrams and bigrams, `max_features = 5000`, `min_df = 2` and sublinear TF. TF-IDF is fitted inside each cross-validation fold (scikit-learn `Pipeline`), and both classifiers are evaluated with 5-fold stratified cross-validation over the 170 documents. A full run takes under five minutes on a laptop CPU.

## 6.2 Tier 2 — Domain-specific fine-tuning (LayoutLMv3)
*Question: does fine-tuning a pre-trained multimodal document transformer outperform the baseline?*

LayoutLMv3 [1], the successor to LayoutLMv2 [4], is a multimodal Transformer [3] (base model: 133M parameters) that jointly encodes (1) text tokens, (2) each word's bounding box normalised to [0, 1000], and (3) image patches of the page. It is pre-trained on large document collections with unified text and image masking; fine-tuning adapts it to the eight ALIP classes. Mineral licensing documents have layout cues — letterheads, licence-number blocks, coordinate tables, signature blocks, labelled form fields — that TF-IDF ignores and LayoutLMv3 can in principle exploit.

*Implementation.* Each PDF page is rendered at 200 DPI with pdf2image and OCR'd with Tesseract [5] for words and boxes, then encoded with the LayoutLMv3 processor (max 512 tokens) via Hugging Face Transformers [6]. Training uses AdamW (lr = 1e-5), batch size 4, a cosine schedule with 10% warm-up, up to 15 epochs, and early stopping on validation macro F1, on a Colab T4 GPU (≈ 5–10 min per run). Tesseract replaced the originally planned PaddleOCR for simpler cross-platform setup; the inputs were born-digital PDFs, where Tesseract is reliable.

## 6.3 Tier 3 — General-purpose LLM reference (Llama 3.1 8B)
*Question: does a general-purpose LLM, with no fine-tuning, already solve the task?*

Llama 3.1 8B [2] is an open-weight model released by Meta in July 2024 and pre-trained on roughly 15 trillion tokens of public data. Its weights run locally through Ollama [8], so no document content leaves the machine and no API fees apply.

In **zero-shot** mode the prompt lists the eight class names with short descriptions. In **few-shot** mode one synthetic training example per class is added. No weights are updated in either mode.

*Implementation.* pdfplumber text truncated to 2,500 characters; temperature 0; responses parsed by exact match, then substring match, then keyword match. CPU inference took ≈ 99 s per document (≈ 4.5 h per mode).

**Table 6: LayoutLMv3 vs. Llama 3.1 8B.**

| Dimension | LayoutLMv3 (Tier 2) | Llama 3.1 8B (Tier 3) |
|---|---|---|
| Input | Text + layout + page image | Text only (first 2,500 chars) |
| Domain adaptation | Fine-tuned on 109 docs | Prompt only |
| Labelled data required | Yes | None (zero-shot) |
| Parameters | 133M | 8B |
| Hardware | Colab T4 GPU | Laptop CPU via Ollama |
| Training cost | $0 (Colab free tier) | $0 |
| Macro F1 | 0.943 | 0.989 (zero-shot) |

# 7. How the Results Were Obtained

## 7.1 Baseline
`baseline_pipeline.py --data <unified_dataset>` extracts and preprocesses text, runs 5-fold stratified CV for both classifiers, and writes per-class metrics, confusion matrices and plots.

## 7.2 LayoutLMv3
1. **Data preparation (local).** `layoutlmv3_data_prep.py` renders pages, runs Tesseract, normalises boxes, encodes with the processor, and saves a Hugging Face Arrow dataset, uploaded to Colab.
2. **Fine-tuning (Colab T4).** `layoutlmv3_finetune.py` loads `microsoft/layoutlmv3-base`, adds an 8-class head and trains with early stopping; the best checkpoint is saved to Google Drive.
3. **Re-evaluation.** `layoutlmv3_reeval.py` loads the checkpoint and predicts on all 170 documents.

## 7.3 Llama 3.1
`llm_evaluation.py` sends each document's text with the class-description prompt to the local Ollama API, parses the label and saves JSON — once zero-shot, once few-shot.

## 7.4 Comparison
`compare_models.py` loads all five result files and computes macro F1, accuracy, precision and recall, producing Tables 7–8.

## 7.5 Evaluation protocol — important caveat
The three tiers were **not** evaluated under an identical protocol:

* **TF-IDF:** out-of-fold predictions from 5-fold CV over all 170 documents — each document is predicted by a model that did not see it.
* **LayoutLMv3:** predictions on all 170 documents from a model trained on 109 and early-stopped on 27 of them. **136 of the 170 evaluation documents were seen during training or model selection**, so its score is optimistic. Even so, it scored lowest.
* **Llama 3.1:** predictions on all 170 documents with no training; few-shot examples were drawn from the synthetic pool.

The comparison that is fully like-for-like — all models on the 34-document held-out test split, and separately on the 10 real documents — has not yet been reported and is the first item of future work.

# 8. Experimental Results

## 8.1 Main results

**Table 7: Macro-averaged metrics, 170 documents (protocols per §7.5).**

| Model | Macro F1 | Accuracy | Precision | Recall | Correct | Tier |
|------------------------------------|---------|---------|---------|-------|-------|----|
| TF-IDF + Logistic Regression | 0.9844 | 0.9824 | 0.9881 | 0.9838 | 167 | 1 |
| TF-IDF + LinearSVC | 0.9844 | 0.9824 | 0.9881 | 0.9838 | 167 | 1 |
| LayoutLMv3 (fine-tuned) | 0.9429 | 0.9412 | 0.9484 | 0.9419 | 160 | 2 |
| Llama 3.1 8B (zero-shot) | 0.9892 | 0.9882 | 0.9886 | 0.9907 | 168 | 3 |
| Llama 3.1 8B (few-shot) | 0.9788 | 0.9765 | 0.9794 | 0.9794 | 166 | 3 |

**Tier 1.** Both classifiers correctly labelled 167 of 170 documents with identical metrics, indicating the representation, not the classifier, is the limiting factor. From the per-class F1 scores and class sizes, the three errors were two real Small-Scale Mining Permits and one real Renewal Application predicted as New License Application — real application-style forms share vocabulary across these classes.

**Tier 2.** LayoutLMv3 labelled 160 of 170 correctly, 4.1 points of macro F1 below TF-IDF, despite the evaluation overlapping its training data. Errors were spread across several classes, including synthetic-only ones (Exploration License F1 = 0.833, Mining Lease, EIA, ECC). *(VERIFY: the per-class error counts against the re-evaluation JSON.)* Two factors likely contribute: ~13–14 synthetic training documents per class is too few to learn layout features that transfer, and a possible input-encoding issue in the processor call (§10.3) that should be ruled out before drawing conclusions about the architecture.

**Tier 3.** Zero-shot Llama labelled 168 of 170 correctly with no labelled data. Both errors were real New License Applications. The per-class scores are consistent with both being predicted as Renewal Application — a plausible confusion between two kinds of application form. *(VERIFY: Table 8 lists Renewal F1 as 0.952 while this pattern gives 0.955.)* Few-shot prompting labelled 166 correctly, two fewer than zero-shot; with differences this small we do not attribute a cause.

**Significance.** Zero-shot Llama and TF-IDF differ by one document out of 170; the difference is not meaningful without a significance test. The defensible conclusions are that (a) both clearly beat LayoutLMv3 as trained here, and (b) the fine-tuning hypothesis was not supported.

## 8.2 Per-class F1

**Table 8: Per-class F1. "Real" = class has real test documents.**

| Class | Support (real) | TF-IDF + LR | LayoutLMv3 | Llama 0-shot | Real? |
|---|---|---|---|---|---|
| Exploration License | 20 (0) | 1.000 | 0.833 | 1.000 | No |
| Prospecting Permit | 20 (0) | 1.000 | 1.000 | 1.000 | No |
| Mining Lease | 20 (0) | 1.000 | 0.976 | 1.000 | No |
| Small-Scale Mining Permit | 22 (2) | 0.952 | 0.952 | 1.000 | **Yes** |
| New License Application | 27 (7) | 0.947 | 0.877 | 0.962 | **Yes** |
| Renewal Application | 21 (1) | 0.976 | 1.000 | 0.952 | **Yes** |
| Env. Impact Assessment | 20 (0) | 1.000 | 0.950 | 1.000 | No |
| Env. Compliance Certificate | 20 (0) | 1.000 | 0.955 | 1.000 | No |
| **Macro average** | 170 (10) | **0.984** | **0.943** | **0.989** | — |

The perfect scores on the five synthetic-only classes reflect the title-level leakage (§5.2) and confirm the pipeline works; they are not evidence of real-world generalisation. Every TF-IDF and Llama error involves a real document in one of the three application/permit classes, which is where the real difficulty lies. Renewal Application is the only class where LayoutLMv3 scores highest (1.000, one real document), too small a sample to conclude that layout awareness helps.

# 9. LLM Implementation: Data Privacy and Sovereignty

## 9.1 Provider data handling

**Table 9: LLM provider data policies (as of the study period).**

| Provider | Consumer products | API / enterprise | ALIP decision |
|---|---|---|---|
| OpenAI | Inputs may be used for training unless the user opts out | Not used for training by default | Synthetic docs only |
| Anthropic | Consumer chats used for training only if the user allows it (2025 policy) | Not used for training | Synthetic docs only |
| Meta Llama (local) | N/A — runs locally | N/A — no data leaves the machine | Used for all real-document experiments |

## 9.2 Confidentiality and copyright
Mineral licensing documents contain commercially sensitive information — licence coordinates, applicant names, financial guarantees and proprietary geological data — and application documents are the intellectual property of their authors. Sending them to an external provider whose terms permit training on inputs raises confidentiality and copyright concerns.

## 9.3 Data sovereignty
Several African Union member states have data-protection laws modelled on the EU GDPR. Transferring government documents to foreign servers can require Data Processing Agreements or Standard Contractual Clauses — a real burden for ministries. All ALIP experiments on real documents used locally run models only; no document content was transmitted externally.

# 10. Challenges, Limitations and Mitigations

**Table 10: Risk register — planned mitigations vs. outcomes.**

| Risk | Planned mitigation | Outcome |
|---|---|---|
| Insufficient real documents | Supplement with synthetic; report limitation | 27 real collected, 10 usable for Tier 1; 5 of 8 classes synthetic-only |
| Class imbalance | Macro-F1; SMOTE if needed | Macro-F1 used; SMOTE unnecessary (balanced generation) |
| OCR quality on scans | Evaluate OCR; Donut as OCR-free fallback | All inputs born-digital; Tesseract adequate; Donut not needed |
| LayoutLMv3 pipeline complexity | Buffer week; DistilBERT fallback | Tesseract replaced PaddleOCR; fallback not needed |
| Label disagreement | Two annotators; Cohen's κ | Synthetic labels from generator; real docs by team consensus (κ not computed) |
| Tier 2 contamination of Tier 1 | Strict manifest labelling | 14 real Tier 2 docs excluded from the 8-class evaluation |
| Synthetic label leakage | *(not anticipated)* | Class titles present in synthetic docs (§5.2) — open |
| Evaluation leakage | *(not anticipated)* | LayoutLMv3 re-evaluated on data it was trained on (§7.5) — open |

**10.1 Colab session management.** Fine-tuning needed repeated Colab sessions; the ~480 MB checkpoint was saved to Google Drive immediately after training. The default `save_total_limit=2` exhausted Drive quota, so it was set to 1.

**10.2 Windows/Linux paths.** The dataset merger, run on Windows, wrote manifests with backslash paths, which fail on Colab. Fixed with `rel.replace('\\', '/')` in `layoutlmv3_reeval.py`.

**10.3 Processor output shape.** At inference, the processor returned `input_ids` of shape `[383, 512]` instead of `[1, 512]`. The workaround kept only the first row (`enc[k][0:1]`). A first dimension in the hundreds is the typical symptom of passing `words` and `boxes` un-nested, so each *word* is treated as a separate example; if so, the workaround classifies each page from its first word only. The correct call is `processor(image, [words], boxes=[boxes], truncation=True, max_length=512, return_tensors="pt")`. This must be checked and LayoutLMv3 re-run before its result is treated as final.

**10.4 Local checkpoint loading.** `from_pretrained(path, local_files_only=True)` was used so Transformers loads the checkpoint from disk instead of treating the path as a Hub repo ID.

# 11. Future Work

* **Like-for-like evaluation.** Report all models on the 34-document held-out split and on the 10 real documents alone; run McNemar's test on paired predictions.
* **Fix and re-run LayoutLMv3** after checking the processor call (§10.3).
* **Remove synthetic leakage.** Regenerate documents without class titles or class-naming phrases and re-run all tiers as an ablation.
* **Expand the real corpus.** Source real documents for the five synthetic-only classes (e.g. Zimbabwe, Zambia MCAS, South Africa DMRE); even 5–10 per class would turn pipeline validation into a generalisation measurement.
* **Larger and alternative LLMs** (e.g. Llama 3.1 70B, Mistral 7B) with structured JSON output for reliable parsing.
* **Full 44-class taxonomy**, including Tier 2 types.
* **System integration:** an intake dashboard that classifies, flags deficiencies and routes documents into cadastre platforms.

# 12. Individual Contributions

**Table 11: Team member contributions.**

| Member | Role | Contributions |
|---|---|---|
| Enock Katenda | Project Lead / Domain Research | Project ideation and framing; regulatory research across five African jurisdictions defining the 8-class and 44-class taxonomies; sourced and classified the 27 real documents; authored the proposal (April 2025), goals document and taxonomy analysis |
| Ezekiel (Simba) Ponde | Technical Lead / ML Implementation | Built and debugged the ML pipeline across all three tiers; resolved Colab/Windows compatibility issues; ran all experiments and produced all quantitative results; authored technical documentation, README and final report |
| Tafadzwa Nyemba | Dataset & Validation | Dataset research and real-document analysis; taxonomy validation and regulatory review; pipeline testing and quality validation; contributed to experimental design and interpretation |

# 13. Conclusion

ALIP implemented and evaluated a three-tier approach to classifying mineral licensing documents in Sub-Saharan African regulatory contexts. The central hypothesis — that fine-tuning LayoutLMv3 would outperform a classical baseline — was not supported: LayoutLMv3 reached macro F1 = 0.943, below TF-IDF (0.984) and zero-shot Llama 3.1 8B (0.989), even though its evaluation overlapped its training data. TF-IDF and zero-shot Llama were effectively tied.

> **Answer to the research question.** In this low-resource setting, domain-specific fine-tuning of LayoutLMv3 did not outperform a TF-IDF baseline. A locally hosted zero-shot LLM matched the baseline with no labelled data and no external data transfer. Because synthetic documents contain class titles, five classes lack real test data, and protocols differ across tiers, these results show feasibility rather than real-world accuracy; the re-runs in §11 are needed to establish generalisation.

# 14. Evaluation Framework

* **Macro F1** — primary metric; weights all classes equally.
* **Accuracy** — reported, not primary.
* **Macro precision** — per-class precision (share of predictions for a class that are correct), averaged equally.
* **Macro recall** — per-class recall (share of a class's documents that are found), averaged equally.
* **Confusion matrix** — systematic error patterns.
* **McNemar's test** — planned for paired significance testing; to be run in the like-for-like re-evaluation.

# References

[1] Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022). LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. *Proc. 30th ACM International Conference on Multimedia.*

[2] Grattafiori, A., Dubey, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems 30.*

[4] Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C., Che, W., Zhang, M., & Zhou, L. (2021). LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding. *Proc. ACL-IJCNLP 2021.*

[5] Smith, R. (2007). An Overview of the Tesseract OCR Engine. *Proc. 9th International Conference on Document Analysis and Recognition (ICDAR).*

[6] Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proc. EMNLP 2020: System Demonstrations.*

[7] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12,* 2825–2830.

[8] Ollama. (2024). Run large language models locally. https://ollama.com

[9] Trimble Inc. (2024). Landfolio mineral licensing and cadastre platform. https://www.trimble.com/en/products/software/landfolio

[10] Revenue Development Foundation. (2024). Mining Cadastre Administration System (MCAS).

[11] U.S. Bureau of Land Management. (2024). Mineral and Land Records System (MLRS). https://www.blm.gov/mlrs

[12] Republic of Uganda. (2022). *The Mining and Minerals Act, 2022.* Ministry of Energy and Mineral Development.

[13] Republic of Botswana. *Mines and Minerals Act* (Cap. 66:01). Ministry of Minerals and Energy.

[14] Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A Library for Large Linear Classification. *Journal of Machine Learning Research, 9,* 1871–1874.
