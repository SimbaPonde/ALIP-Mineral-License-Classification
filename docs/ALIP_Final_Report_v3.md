---
title: "ALIP – Aquila Land Intelligence Platform"
subtitle: "Mineral License Document Classification: A Three-Tier Experimental Study (Revised v3)"
author:
  - "Enock Katenda · Ezekiel Ponde · Tafadzwa Nyemba"
  - "Katz School of Science and Health, Yeshiva University, New York, NY"
date: "Capstone Project — May 2026 (revised and re-evaluated November 2026)"
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

> **Key result.** On a held-out test set of 34 documents — including all 10 real government forms, never seen in training — few-shot Llama 3.1 8B scored macro F1 = 0.957 (8/10 real correct), TF-IDF 0.929 (7/10) and fine-tuned LayoutLMv3 0.769 (1/10). Fine-tuning learned the layouts of the synthetic templates and did not transfer to real forms. The fine-tuning hypothesis was not supported.

## Abstract

ALIP investigates whether domain-specific fine-tuning measurably improves Intelligent Document Processing (IDP) for mineral licensing documents in a low-resource African regulatory context. We build a dataset of 170 documents — 160 synthetic PDFs across eight Tier 1 classes and 10 real government forms from Botswana and Uganda (drawn from 27 collected) — with a fixed split in which real documents appear only in the 34-document test set. Three tiers are compared on identical documents: (1) TF-IDF with Logistic Regression / LinearSVC; (2) fine-tuned LayoutLMv3-base [1]; and (3) Llama 3.1 8B [2] in zero-shot and few-shot modes, run locally via Ollama [8]. All models classify nearly every synthetic document correctly; real forms separate them. Few-shot Llama achieved the best macro F1 (0.957; 8/10 real), followed by TF-IDF (0.929; 7/10) and zero-shot Llama (0.919; 8/10). Fine-tuned LayoutLMv3 scored 0.769 and classified only 1 of 10 real documents correctly — significantly worse than few-shot Llama (exact McNemar p = 0.039). With roughly 14 template-generated training documents per class, the layout-aware model memorised template layouts rather than learning transferable document semantics. A class-name masking ablation left TF-IDF unchanged. The practical conclusion: for this task, a locally hosted LLM with a handful of examples, or a cheap lexical baseline, outperforms layout-aware fine-tuning on synthetic data, while keeping sensitive documents on-premises. This revision (v3) replaces the original, non-comparable evaluation (§8.4).

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
| Reproducible TF-IDF + LR/SVM baseline | Achieved | Macro F1 = 0.929 held-out; 7/10 real |
| Fine-tune LayoutLMv3 and compare | Achieved | Macro F1 = 0.769 held-out; 1/10 real |
| Accuracy, precision, recall, macro-F1 | Achieved | All models, identical test documents, McNemar tests |
| Document impact of synthetic data | Achieved | Synthetic test accuracy ≈ 100% for all models; real accuracy 1–8/10 |
| General-purpose LLM reference | Achieved | Llama 3.1 8B few-shot F1 = 0.957; zero-shot 0.919 |

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

**Limitation — label leakage.** Each template prints its class name as the document title (e.g. "EXPLORATION LICENSE", "APPLICATION FOR RENEWAL OF …"), and body text contains class-specific phrases ("hereby grants an Exploration License"). A classifier could succeed on synthetic documents by keyword lookup. A masking ablation (§7.2) removes all class-naming phrases; TF-IDF still classifies every synthetic test document correctly, so the templates are separable even without them. Synthetic accuracy therefore says little about real-world performance, which is why real documents are reported separately throughout §8.

## 5.3 Real document collection
27 real government mineral licensing documents were collected from public regulatory portals: 4 from Botswana's Ministry of Mineral Resources [13] and 23 from Uganda's licensing portal under the Mining and Minerals Act 2022 [12]. Of these, 10 map to Tier 1 classes and are used as test documents; 14 map to Tier 2 classes and are excluded from the 8-class experiments; the remaining 3 (Uganda Forms 36, 39, 41 — mineral processing, smelting and refining licences) fall outside the taxonomy. Labels are listed in `data/real_labels.csv`. **All 10 real Tier 1 documents are blank application forms**; the dataset contains no real issued licence, lease, EIA or compliance certificate.

> **Evaluation note.** Five of the eight classes — Exploration License, Prospecting Permit, Mining Lease, EIA and Environmental Compliance Certificate — have no real documents. Scores on those classes validate the pipeline on synthetic data only. The three classes with real documents (New License Application, Small-Scale Mining Permit, Renewal Application) carry the substantive evaluation.

**Table 5: Fixed dataset split used by all models.**

| Split | Total | Real | Synthetic | Notes |
|---|---|---|---|---|
| Train | 109 | 0 | 109 | Synthetic only |
| Validation | 27 | 0 | 27 | Synthetic only — LayoutLMv3 checkpoint selection |
| Test | 34 | 10 | 24 | 3 synthetic per class + all 10 real; used for every reported result |
| **Total** | **170** | **10** | **160** | |

# 6. Methodology and Models

## 6.1 Tier 1 — Classical baseline (TF-IDF + LR / SVM)
*Question: what is the performance floor for lightweight, interpretable models?*

TF-IDF represents each document as a sparse vector in which a term's weight rises with its frequency in the document and falls with the number of documents containing it, so frequent-but-ubiquitous words carry little weight. **Logistic Regression** learns a weight per term per class and converts the class scores to probabilities with a softmax. It is well suited to settings where features greatly outnumber training examples, as here. **LinearSVC** fits maximum-margin linear boundaries, one-vs-rest across the eight classes [14]. Both are implemented in scikit-learn [7].

*Implementation.* Text is extracted with pdfplumber and preprocessed with NLTK (lower-casing, stop-word removal, WordNet lemmatisation). The vectoriser uses unigrams and bigrams, `max_features = 5000`, `min_df = 2` and sublinear TF. The vectoriser and classifier are fitted on the train split only (scikit-learn `Pipeline`). A full run takes about a minute on a laptop CPU.

## 6.2 Tier 2 — Domain-specific fine-tuning (LayoutLMv3)
*Question: does fine-tuning a pre-trained multimodal document transformer outperform the baseline?*

LayoutLMv3 [1], the successor to LayoutLMv2 [4], is a multimodal Transformer [3] (base model: 133M parameters) that jointly encodes (1) text tokens, (2) each word's bounding box normalised to [0, 1000], and (3) image patches of the page. It is pre-trained on large document collections with unified text and image masking; fine-tuning adapts it to the eight ALIP classes. Mineral licensing documents have layout cues — letterheads, licence-number blocks, coordinate tables, signature blocks, labelled form fields — that TF-IDF ignores and LayoutLMv3 can in principle exploit.

*Implementation.* Each PDF page is rendered at 200 DPI with pdf2image and OCR'd with Tesseract [5] for words and boxes, then encoded with the LayoutLMv3 processor (max 512 tokens) via Hugging Face Transformers [6]. Training uses AdamW (lr = 1e-5), batch size 4, a cosine schedule with 10% warm-up and up to 15 epochs on a Colab T4 GPU (≈ 6 min); checkpoint selection is described in §7.3. Tesseract replaced the originally planned PaddleOCR for simpler cross-platform setup; the inputs were born-digital PDFs, where Tesseract is reliable.

## 6.3 Tier 3 — General-purpose LLM reference (Llama 3.1 8B)
*Question: does a general-purpose LLM, with no fine-tuning, already solve the task?*

Llama 3.1 8B [2] is an open-weight model released by Meta in July 2024 and pre-trained on roughly 15 trillion tokens of public data. Its weights run locally through Ollama [8], so no document content leaves the machine and no API fees apply.

In **zero-shot** mode the prompt lists the eight class names with short descriptions. In **few-shot** mode one synthetic training example per class is added. No weights are updated in either mode.

*Implementation.* pdfplumber text truncated to 2,500 characters; temperature 0; output constrained to the eight class IDs by a JSON schema, with exact/substring/keyword parsing as fallback. Inference took ≈ 99 s per document on a laptop CPU and 1–4 s on a Colab T4 GPU.

**Table 6: LayoutLMv3 vs. Llama 3.1 8B.**

| Dimension | LayoutLMv3 (Tier 2) | Llama 3.1 8B (Tier 3) |
|---|---|---|
| Input | Text + layout + page image | Text only (first 2,500 chars) |
| Domain adaptation | Fine-tuned on 109 docs | Prompt only |
| Labelled data required | Yes (109 docs) | None (zero-shot) / 8 excerpts (few-shot) |
| Parameters | 133M | 8B |
| Hardware | Colab T4 GPU | Ollama on laptop CPU or Colab T4 |
| Training cost | $0 (Colab free tier) | $0 |
| Held-out macro F1 | 0.769 (1/10 real) | 0.957 few-shot / 0.919 zero-shot (8/10 real) |

# 7. How the Results Were Obtained

The original scripts were lost after the May 2026 submission and the pipeline was rebuilt in November 2026 (repository: `github.com/SimbaPonde/ALIP-Mineral-License-Classification`). The rebuild fixed the protocol problems of the original evaluation, so **every model is now scored on the same held-out documents**.

## 7.1 Unified dataset and split
`build_unified_dataset.py` merges the 160 synthetic PDFs with the 10 real Tier 1 forms and writes one manifest with a fixed, stratified split (seed 42): **train** 109 synthetic, **validation** 27 synthetic, **test** 34 (3 synthetic per class + all 10 real). Real documents appear only in the test split, so no model is trained, tuned or prompted with them. Paths are stored repo-relative with forward slashes (fixing §10.2 at the source).

## 7.2 Tier 1 — TF-IDF
`baseline_heldout.py` fits TF-IDF + LR and TF-IDF + LinearSVC on the train split and predicts the test split. A label-leakage ablation (`--mask-class-names`) deletes every class-naming phrase ("exploration licence", "renewal", "permit", "certificate", …) from all documents before vectorising.

## 7.3 Tier 2 — LayoutLMv3
`layoutlmv3_pipeline.py` (run on a Colab T4): Tesseract OCR of page 1 → `microsoft/layoutlmv3-base` fine-tuned for up to 15 epochs (AdamW, lr 1e-5, batch 4, cosine schedule, 10% warm-up) → prediction on the test split. Two fixes relative to the original: the processor receives words and boxes as a single example, with an assertion that inputs are `[1, 512]` (§10.3); and because validation macro-F1 reaches 1.000 by epoch 2 on synthetic data, checkpoint selection breaks F1 ties by validation loss. The selected checkpoint is from epoch 15 (validation loss 0.030). Training takes ≈ 6 min.

## 7.4 Tier 3 — Llama 3.1 8B
`llm_evaluation.py` queries Llama 3.1 8B through a local Ollama server: temperature 0, first 2,500 characters of pdfplumber text, output constrained by a JSON schema to the eight class IDs (all 68 responses parsed as valid JSON). Few-shot mode adds one 700-character excerpt per class, drawn from the train split. On a T4 GPU inference took 1–4 s per document.

## 7.5 Comparison
`compare_models.py` scores all prediction files on the identical 34 documents: macro F1, accuracy, macro precision and recall, accuracy on the 10 real documents alone, per-class F1, and pairwise exact McNemar tests.

# 8. Experimental Results

## 8.1 Main results (held-out test split)

**Table 7: All models on the same 34 held-out documents (24 synthetic + 10 real).**

| Model | Macro F1 | Acc. | Prec. | Rec. | Correct | Real | Tier |
|----------------------------------------|---------|------|------|------|--------|------|----|
| TF-IDF + Logistic Regression | 0.929 | 0.912 | 0.948 | 0.919 | 31/34 | 7/10 | 1 |
| TF-IDF + LinearSVC | 0.929 | 0.912 | 0.948 | 0.919 | 31/34 | 7/10 | 1 |
| TF-IDF + LR, class names masked | 0.929 | 0.912 | 0.948 | 0.919 | 31/34 | 7/10 | 1 |
| LayoutLMv3 (fine-tuned) | 0.769 | 0.735 | 0.765 | 0.856 | 25/34 | 1/10 | 2 |
| Llama 3.1 8B (zero-shot) | 0.919 | 0.882 | 0.929 | 0.950 | 30/34 | 8/10 | 3 |
| **Llama 3.1 8B (few-shot)** | **0.957** | **0.941** | **0.979** | **0.950** | **32/34** | **8/10** | 3 |

**Every model classifies all 24 synthetic test documents correctly** except zero-shot Llama (22/24). The models are therefore separated almost entirely by the 10 real government forms.

**Tier 1.** TF-IDF misclassifies 3 real forms: Uganda Forms 19 and 21 (small-scale and artisanal mining licence *applications*, labelled Small-Scale Mining Permit) are predicted as New License Application, and Form 25 (renewal) is predicted as Small-Scale Mining Permit. LR and LinearSVC make identical predictions. Masking class names changes nothing: the synthetic templates are separable by their remaining vocabulary, so title leakage is not what drives the synthetic scores.

**Tier 2.** LayoutLMv3 gets all 24 synthetic documents right and **only 1 of 10 real documents**. Its real-document predictions are scattered across six classes (e.g. a Uganda prospecting-licence application predicted as Exploration License, a Botswana prospecting-licence application as Environmental Compliance Certificate). The model has learned the page layouts of the eight synthetic templates; real forms match none of them. With ≈ 14 training documents per class, all generated from one template each, layout is the easiest feature to fit and the one that transfers worst. An earlier run that kept the barely-trained epoch-2 checkpoint scored similarly (26/34), so the failure is not a checkpoint-selection artefact.

**Tier 3.** Zero-shot Llama labels applications *for* a prospecting licence as Prospecting Permit (4 errors: 2 synthetic, 2 real) — it confuses an application with the instrument applied for. One example per class (few-shot) removes this confusion. Few-shot Llama's only errors are the two small-scale/artisanal licence applications, predicted as New License Application — the same two documents TF-IDF misses, and arguably a labelling ambiguity rather than a model error (§10).

## 8.2 Statistical significance

**Table 8: Exact McNemar tests on the 34 paired predictions (selected pairs).**

| Model A | Model B | Only A correct | Only B correct | p |
|---|---|---|---|---|
| Llama few-shot | LayoutLMv3 | 8 | 1 | **0.039** |
| TF-IDF + LR | LayoutLMv3 | 7 | 1 | 0.070 |
| Llama zero-shot | LayoutLMv3 | 7 | 2 | 0.180 |
| Llama few-shot | TF-IDF + LR | 1 | 0 | 1.000 |
| Llama few-shot | Llama zero-shot | 4 | 2 | 0.688 |
| TF-IDF + LR | Llama zero-shot | 4 | 3 | 1.000 |

Only one difference is significant at α = 0.05: few-shot Llama outperforms fine-tuned LayoutLMv3. TF-IDF and the two Llama modes cannot be separated on 34 documents.

## 8.3 Per-class F1

**Table 9: Per-class F1 on the test split. Support: 3 synthetic per class, plus the real documents shown.**

| Class | Real in test | TF-IDF + LR | LayoutLMv3 | Llama 0-shot | Llama few-shot |
|---|---|---|---|---|---|
| Exploration License | 0 | 1.000 | 0.750 | 1.000 | 1.000 |
| Prospecting Permit | 0 | 1.000 | 0.857 | 0.600 | 1.000 |
| Mining Lease | 0 | 1.000 | 0.857 | 1.000 | 1.000 |
| Small-Scale Mining Permit | 2 | 0.667 | 0.727 | 1.000 | 0.750 |
| New License Application | 7 | 0.909 | 0.462 | 0.750 | 0.909 |
| Renewal Application | 1 | 0.857 | 0.750 | 1.000 | 1.000 |
| Env. Impact Assessment | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Env. Compliance Certificate | 0 | 1.000 | 0.750 | 1.000 | 1.000 |
| **Macro average** | 10 | **0.929** | **0.769** | **0.919** | **0.957** |

LayoutLMv3's scores below 1.000 on synthetic-only classes come from false positives — real forms predicted into those classes — not from missed synthetic documents.

## 8.4 Original results (May 2026)

The original submission reported macro F1 of 0.984 (TF-IDF, 5-fold CV over 170 documents), 0.943 (LayoutLMv3, evaluated on 170 documents including 136 it was trained or tuned on, with the processor bug of §10.3) and 0.989 / 0.979 (Llama zero-/few-shot on 170 documents). Because the three protocols differed and 160 of the 170 documents were synthetic, those figures overstated real-world accuracy — most sharply for LayoutLMv3, whose real-document accuracy is 1 in 10 once evaluated on held-out data. Table 7 supersedes them.


# 9. LLM Implementation: Data Privacy and Sovereignty

## 9.1 Provider data handling

**Table 10: LLM provider data policies (as of the study period).**

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

**Table 11: Risk register — planned mitigations vs. outcomes.**

| Risk | Planned mitigation | Outcome |
|---|---|---|
| Insufficient real documents | Supplement with synthetic; report limitation | 27 real collected, 10 usable for Tier 1; 5 of 8 classes synthetic-only |
| Class imbalance | Macro-F1; SMOTE if needed | Macro-F1 used; SMOTE unnecessary (balanced generation) |
| OCR quality on scans | Evaluate OCR; Donut as OCR-free fallback | All inputs born-digital; Tesseract adequate; Donut not needed |
| LayoutLMv3 pipeline complexity | Buffer week; DistilBERT fallback | Tesseract replaced PaddleOCR; fallback not needed |
| Label disagreement | Two annotators; Cohen's κ | Synthetic labels from generator; real labels by consensus (κ not computed). Small-scale *applications* labelled Small-Scale Mining Permit are the main ambiguity — every model's errors include them |
| Tier 2 contamination of Tier 1 | Strict manifest labelling | 14 real Tier 2 docs excluded from the 8-class evaluation |
| Synthetic label leakage | *(not anticipated)* | Class titles in synthetic docs; masking ablation shows no effect on TF-IDF (§8.1) |
| Evaluation leakage | *(not anticipated)* | Original LayoutLMv3 score used training data; fixed by the held-out split (§7.1) |

**10.1 Colab session management.** Fine-tuning needed repeated Colab sessions; the ~480 MB checkpoint was saved to Google Drive immediately after training. The default `save_total_limit=2` exhausted Drive quota, so it was set to 1.

**10.2 Windows/Linux paths.** The dataset merger, run on Windows, wrote manifests with backslash paths, which fail on Colab. Originally patched with `rel.replace('\\', '/')`; the rebuilt manifest stores forward-slash, repo-relative paths.

**10.3 Processor output shape.** In the original pipeline the processor returned `input_ids` of shape `[383, 512]` and the workaround kept row 0. This is the symptom of passing `words` and `boxes` un-nested, so each word was treated as a separate example and pages were classified from their first word. The rebuilt pipeline passes `[words]`, `[boxes]` and asserts a `[1, 512]` shape. Even with the fix, LayoutLMv3 does not transfer to real forms (§8.1).

**10.5 Checkpoint selection.** Validation macro-F1 saturates at 1.000 by epoch 2 on synthetic data, so selecting on F1 alone kept a barely-trained model. Ties are now broken by validation loss.

**10.4 Local checkpoint loading.** `from_pretrained(path, local_files_only=True)` was used so Transformers loads the checkpoint from disk instead of treating the path as a Hub repo ID.

# 11. Future Work

* **More real data — the binding constraint.** Source real issued licences, leases, EIAs and compliance certificates for the five synthetic-only classes (e.g. Zimbabwe, Zambia MCAS, South Africa DMRE). Even 5–10 per class would allow real-document training and a meaningful test.
* **Clarify the application/instrument boundary.** Decide whether an application *for* a small-scale licence belongs to Small-Scale Mining Permit or New License Application; this single labelling decision accounts for most remaining errors.
* **More diverse synthetic layouts.** Several templates per class, randomised field order and real-form-style tables, so layout-aware models cannot memorise one template per class.
* **Larger and alternative LLMs** (Llama 3.1 70B, Qwen, Mistral) and more few-shot examples per class.
* **Full 44-class taxonomy**, including Tier 2 types (14 real Tier 2 forms are already collected).
* **System integration:** an intake dashboard that classifies, flags deficiencies and routes documents into cadastre platforms.

# 12. Individual Contributions

**Table 12: Team member contributions.**

| Member | Role | Contributions |
|---|---|---|
| Enock Katenda | Project Lead / Domain Research | Project ideation and framing; regulatory research across five African jurisdictions defining the 8-class and 44-class taxonomies; sourced and classified the 27 real documents; authored the proposal (April 2025), goals document and taxonomy analysis |
| Ezekiel (Simba) Ponde | Technical Lead / ML Implementation | Built and debugged the ML pipeline across all three tiers; rebuilt the pipeline and held-out evaluation (Nov 2026); ran all experiments and produced all quantitative results; authored technical documentation, README and final report |
| Tafadzwa Nyemba | Dataset & Validation | Dataset research and real-document analysis; taxonomy validation and regulatory review; pipeline testing and quality validation; contributed to experimental design and interpretation |

# 13. Conclusion

ALIP implemented and compared three approaches to classifying mineral licensing documents in Sub-Saharan African regulatory contexts, on a shared held-out test set containing every real document. The central hypothesis — that fine-tuning LayoutLMv3 would outperform a classical baseline — was not supported. Fine-tuned LayoutLMv3 reached macro F1 = 0.769 and classified 1 of 10 real forms correctly; TF-IDF reached 0.929 (7/10), and few-shot Llama 3.1 8B 0.957 (8/10), significantly better than LayoutLMv3 (p = 0.039).

> **Answer to the research question.** No. With ~14 template-generated training documents per class, domain-specific fine-tuning of a layout-aware model memorised the synthetic layouts and failed on real forms. A locally hosted LLM given one example per class performed best, needs no training, and keeps confidential documents on-premises; a TF-IDF baseline was statistically indistinguishable from it. With only 10 real test documents, all of them application forms, these results establish a ranking and a failure mode, not production accuracy — more real data (§11) is the next step.

# 14. Evaluation Framework

* **Macro F1** — primary metric; weights all classes equally.
* **Accuracy** — reported, not primary.
* **Macro precision** — per-class precision (share of predictions for a class that are correct), averaged equally.
* **Macro recall** — per-class recall (share of a class's documents that are found), averaged equally.
* **Confusion matrix** — systematic error patterns.
* **McNemar's test** — exact (binomial) paired test on the 34 shared test documents (§8.2).
* **Real-document accuracy** — reported separately, because synthetic accuracy is near 100% for every model.

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
