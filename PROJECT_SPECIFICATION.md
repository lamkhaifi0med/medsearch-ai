# MedSearch AI — Project Specification

**Document Type:** Master Design Specification
**Version:** 1.0
**Status:** Approved for Development
**Audience:** AI Engineers, Backend Engineers, Frontend Engineers, DevOps, Product, Clinical Advisors
**Classification:** Internal — Pre-Development Design Document

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Vision & Product Positioning](#2-vision--product-positioning)
3. [Problem Statement & Project Goals](#3-problem-statement--project-goals)
4. [Guiding Principles & Non-Goals](#4-guiding-principles--non-goals)
5. [AI Feature Catalogue](#5-ai-feature-catalogue)
6. [End-to-End System Behavior](#6-end-to-end-system-behavior)
7. [Core Modules](#7-core-modules)
8. [The AI Pipeline](#8-the-ai-pipeline)
9. [Supported Document Types](#9-supported-document-types)
10. [Database Design](#10-database-design)
11. [Vector Database Design](#11-vector-database-design)
12. [Embedding Pipeline](#12-embedding-pipeline)
13. [LLM Module](#13-llm-module)
14. [Retrieval Pipeline](#14-retrieval-pipeline)
15. [Medical AI & Clinical NLP](#15-medical-ai--clinical-nlp)
16. [Frontend Specification](#16-frontend-specification)
17. [User Flow](#17-user-flow)
18. [Technology Stack](#18-technology-stack)
19. [Software Architecture](#19-software-architecture)
20. [Folder Structure](#20-folder-structure)
21. [Development Roadmap](#21-development-roadmap)
22. [Evaluation & Quality Strategy](#22-evaluation--quality-strategy)
23. [Security, Privacy & Compliance](#23-security-privacy--compliance)
24. [Non-Functional Requirements](#24-non-functional-requirements)
25. [Risk Register](#25-risk-register)
26. [Future Features](#26-future-features)
27. [Glossary](#27-glossary)

---

# 1. Executive Summary

MedSearch AI is an intelligent clinical case retrieval platform designed to help healthcare professionals find medically similar patient cases from millions of historical records. It combines semantic search, medical embeddings, vector databases, hybrid retrieval, and Retrieval-Augmented Generation (RAG) with Large Language Models to transform unstructured clinical archives into an instantly searchable, explainable evidence base.

The platform is explicitly a **Clinical Decision Support System (CDSS)**. It does **not** diagnose patients, recommend treatments autonomously, or replace clinical judgment. Its single responsibility is to **retrieve prior evidence** — similar historical cases, their treatments, their outcomes, and relevant medical literature — and to **explain, transparently and with citations, why each retrieved case is relevant** to the patient currently under review. The physician remains the sole decision-maker at every step.

From an engineering standpoint, MedSearch AI is a production-grade AI system, not an academic prototype. It is designed around the following pillars:

- **A multi-stage AI pipeline**: OCR → cleaning → medical entity extraction → normalization → embedding → vector indexing → hybrid retrieval → reranking → context construction → grounded LLM reasoning → structured explanation.
- **A polyglot persistence layer**: PostgreSQL for structured clinical and operational data, Qdrant for dense/sparse vector search, Redis for caching, queues, and session state.
- **Explainability as a first-class feature**: every similarity result ships with a confidence score, a factor-by-factor similarity breakdown, and source citations pointing back to the exact passages in the retrieved records.
- **Safety by design**: strict grounding of all LLM output in retrieved evidence, hallucination detection, refusal behaviors for diagnostic requests, comprehensive audit logging, and role-based access control suitable for healthcare environments.
- **Operational maturity**: containerized deployment, background workers for heavy AI workloads, observability, evaluation dashboards, and reproducible retrieval-quality benchmarks.

This document is the master specification. Every subsequent design decision, implementation task, and review must trace back to a requirement described here.

---

# 2. Vision & Product Positioning

## 2.1 Vision Statement

> _Every physician should be able to ask: "Show me patients like this one, what was done for them, and how did it go?" — and receive an evidence-backed, explainable answer in seconds instead of never._

Hospitals are sitting on decades of institutional clinical experience locked inside unstructured notes, scanned documents, lab reports, and discharge summaries. That experience is effectively write-only: it is recorded, archived, and almost never retrieved at the moment of care. MedSearch AI turns that dormant archive into a living clinical memory.

## 2.2 Product Positioning

| Dimension          | Positioning                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| Category           | Clinical Decision Support System (CDSS) — evidence retrieval                        |
| Primary users      | Physicians, clinical researchers, hospital data teams                               |
| Secondary users    | Hospital administrators, quality/outcomes analysts, medical educators               |
| Core value         | Semantic retrieval of similar historical cases with transparent, cited explanations |
| Explicit non-value | Diagnosis, treatment prescription, autonomous clinical decisions                    |
| Deployment model   | Self-hosted (hospital infrastructure) or private cloud; Docker-first                |
| Regulatory posture | Decision _support_, human-in-the-loop, full audit trail, de-identification aware    |

## 2.3 Why Now

Three technology shifts make this product feasible today when it was not five years ago:

1. **Medical-capable embedding models.** Multilingual, long-context embedding models (BGE-M3 class) capture clinical semantics — "MI", "myocardial infarction", and "heart attack" land near each other in vector space without hand-built synonym lists.
2. **Production-grade vector databases.** Qdrant provides filtered approximate nearest-neighbor search over millions of vectors with millisecond latency, payload filtering, and hybrid dense+sparse scoring.
3. **Groundable LLMs.** Modern LLMs (Gemini class) can be constrained to reason _only_ over supplied context, produce structured output, and cite their sources — which converts them from a hallucination risk into an explanation engine.

## 2.4 What Success Looks Like

- A physician uploads a de-identified case summary and receives the top-10 most similar historical cases in under 5 seconds, each with a similarity explanation and confidence score.
- A researcher asks in natural language: _"female patients over 60 with type 2 diabetes who developed acute kidney injury after contrast imaging"_ and receives a precisely filtered, semantically ranked cohort.
- An outcomes analyst compares two treatment paths across the retrieved cohort and exports a cited report.
- The evaluation dashboard shows retrieval precision, reranking gains, groundedness scores, and latency percentiles trending over time — proving the system works, not just claiming it.

---

# 3. Problem Statement & Project Goals

## 3.1 The Problem

Hospitals contain millions of patient records. When a new patient arrives, physicians frequently recall having treated similar patients before — _"I've seen this constellation of symptoms in an elderly patient on anticoagulants…"_ — but cannot efficiently retrieve those records. The knowledge exists; the retrieval mechanism does not.

The obstacles are structural:

1. **Unstructured data dominates.** The clinically richest information lives in free-text notes, scanned faxes, dictated summaries, and heterogeneous lab report formats. Structured fields (ICD codes, demographics) capture only a thin slice of the clinical picture.
2. **Keyword search fails on medical language.** Clinical text is dense with abbreviations ("SOB", "c/o CP", "hx of HTN"), synonyms (heart attack / MI / STEMI / myocardial infarction), negations ("denies chest pain"), and misspellings from OCR. A keyword engine cannot distinguish "no evidence of pneumonia" from "evidence of pneumonia," nor match "renal insufficiency" against "CKD stage 3."
3. **Similarity is multi-dimensional.** Two patients are clinically similar along many axes simultaneously — presentation, comorbidities, medications, lab trajectories, procedures, outcomes. No SQL query expresses "find patients whose overall clinical picture resembles this one."
4. **Retrieval without explanation is useless in medicine.** A black-box "similar patient" list is clinically unacceptable. Physicians need to know _why_ a case was retrieved before they will trust it, and they need pointers back to the source evidence.

## 3.2 Project Goals

For every new patient, the platform shall:

- **G1 — Understand the patient's medical history**: extract and normalize past conditions, surgeries, family history, and risk factors from free text and structured inputs.
- **G2 — Understand symptoms**: recognize presenting complaints, their negation status, severity, and temporality.
- **G3 — Understand diagnoses**: identify diagnosis mentions and map them to ICD-10 and SNOMED CT concepts.
- **G4 — Understand medications**: extract drug names (brand and generic), dosages, routes, and map them to RxNorm concepts.
- **G5 — Understand laboratory results**: parse lab panels, map analytes to LOINC, capture values, units, and abnormality flags.
- **G6 — Understand procedures**: recognize surgical and diagnostic procedures and their timing.
- **G7 — Understand outcomes**: capture disposition, recovery status, complications, readmissions, and mortality signals where documented.
- **G8 — Retrieve the most similar historical patients** using dense semantic search, sparse lexical search, and metadata filtering combined into a single hybrid retrieval pipeline with reranking.
- **G9 — Explain every retrieval.** For each returned case, generate a grounded, cited explanation of _why_ it is similar, which treatments were applied, how the patient responded, and which treatment paths produced better outcomes across the retrieved cohort.
- **G10 — Never diagnose.** The system surfaces evidence; it does not produce diagnostic or prescriptive statements. This constraint is enforced at the prompt, output-validation, and UI layers.

## 3.3 Measurable Objectives

| Objective                                          | Target                                                                                          |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| End-to-end query latency (upload → ranked results) | p95 < 5 s for search over 1M+ indexed cases                                                     |
| Retrieval quality                                  | Recall@10 ≥ 0.85 on the internal evaluation set; nDCG@10 ≥ 0.80 after reranking                 |
| Explanation groundedness                           | ≥ 95% of explanation sentences attributable to retrieved context (automated groundedness check) |
| Entity extraction quality                          | F1 ≥ 0.85 on diagnoses, medications, and lab entities against the annotated gold set            |
| Ingestion throughput                               | ≥ 50 documents/minute/worker for text PDFs; ≥ 10/minute/worker for scanned PDFs (OCR path)      |
| Availability                                       | 99.5% for the search API                                                                        |
| Auditability                                       | 100% of searches, retrievals, and LLM generations logged with immutable audit entries           |

---

# 4. Guiding Principles & Non-Goals

## 4.1 Guiding Principles

1. **Evidence, not answers.** The system's output is always _retrieved evidence plus explanation_, never a clinical conclusion. Language in the UI and LLM output must consistently use retrieval framing ("similar cases show…", "in 7 of 10 retrieved cases…") and never diagnostic framing ("the patient has…", "you should prescribe…").
2. **Explainability is not optional.** Every ranked result must be decomposable: which factors contributed to similarity, what the confidence is, and where in the source documents the supporting text lives.
3. **Grounding over fluency.** An LLM sentence that cannot be traced to retrieved context is a defect, not a feature. The system prefers "insufficient evidence in retrieved cases" over a plausible-sounding fabrication.
4. **Human-in-the-loop always.** Physicians review, accept, dismiss, and annotate results. Those signals feed the evaluation loop.
5. **Privacy by architecture.** De-identification is a pipeline stage, not an afterthought. PHI handling, access control, and audit logging are designed in from day one.
6. **Everything is measurable.** Retrieval quality, extraction quality, explanation groundedness, latency, and cost are tracked continuously in the evaluation dashboard. If it isn't measured, it isn't done.
7. **Boring infrastructure, ambitious AI.** Proven components (PostgreSQL, Redis, Docker, FastAPI, React) carry the operational load so that innovation risk is concentrated where it belongs: the retrieval and reasoning pipeline.

## 4.2 Explicit Non-Goals

- **No diagnosis or treatment recommendation.** The system will refuse prompts asking it to diagnose.
- **No real-time EHR write-back.** MedSearch AI reads clinical data; it never modifies source systems of record.
- **No patient-facing interface.** Users are healthcare professionals and authorized staff only.
- **No autonomous actions.** The system never triggers orders, alerts to patients, or clinical workflows on its own.
- **No training of models on customer PHI** without an explicit, separate, opt-in data governance agreement (out of scope for v1; v1 uses pretrained models plus retrieval).

---

# 5. AI Feature Catalogue

Each feature below is a user-visible capability backed by specific pipeline components described later in this document.

## 5.1 Semantic Search

Free-text queries and whole patient records are embedded into a shared vector space and matched by meaning rather than keywords. "Elderly man with crushing chest pain radiating to the left arm" retrieves myocardial infarction cases even if none of those words appear verbatim in the archive. Backed by the embedding pipeline (§12) and dense retrieval (§14.1).

## 5.2 Medical Embeddings

All clinical text is encoded with BGE-M3, a long-context, multilingual embedding model producing dense vectors, sparse lexical weights, and multi-vector (ColBERT-style) representations from a single forward pass. Section-aware embedding (symptoms embedded separately from medications, labs, history) enables per-facet similarity scoring. Detailed in §12.

## 5.3 Vector Search

Approximate nearest-neighbor search over millions of case vectors in Qdrant using HNSW indexes with cosine similarity, returning ranked candidates in tens of milliseconds. Detailed in §11.

## 5.4 Hybrid Search

Dense (semantic) and sparse (lexical/BM25-style) retrieval run in parallel and are fused (Reciprocal Rank Fusion and/or server-side hybrid scoring). Dense retrieval captures meaning; sparse retrieval protects exact matches that matter in medicine — drug names, gene variants, lab analyte codes, rare disease names. Detailed in §14.3.

## 5.5 Metadata Filtering

Every vector carries a structured payload (age band, sex, ICD-10 codes, encounter year, department, outcome class, record type). Filters are applied _inside_ the vector search (pre-filtering), so "similar cases, but only females over 60 with CKD" is a single filtered ANN query, not a post-hoc cull. Detailed in §11.6 and §14.6.

## 5.6 Retrieval-Augmented Generation (RAG)

The LLM never answers from parametric memory. Retrieved case passages, structured entity summaries, and literature snippets are assembled into a context window; the LLM reasons strictly over that context to produce explanations, comparisons, and summaries. Detailed in §13.

## 5.7 Explainable AI

Every result includes: (a) an overall similarity score, (b) a per-facet breakdown (symptom similarity, diagnosis overlap, medication overlap, lab pattern similarity, demographic proximity), (c) matched-entity highlighting, and (d) an LLM-generated natural-language rationale with citations to source passages. Detailed in §13 and §16.5.

## 5.8 Confidence Scores

Calibrated confidence accompanies each retrieval and each explanation. Confidence is derived from retrieval score distributions (margin between candidates, absolute similarity, reranker score) and groundedness checks on generated text. Low-confidence results are visually flagged and ranked with explicit uncertainty labels ("weak match — retrieved on medication overlap only").

## 5.9 Medical Timeline Generation

The NLP pipeline extracts temporal expressions and anchors clinical events (symptom onset, diagnoses, procedures, medication starts/stops, lab results, outcomes) onto a normalized timeline per patient. The frontend renders an interactive chronological view; the retrieval engine can compare _trajectories_, not just snapshots. Detailed in §7.16 and §15.11.

## 5.10 Natural Language Search

Physicians query in plain language. A query-understanding step extracts structured constraints (demographics, codes, date ranges, outcome filters) from the query, applies them as metadata filters, and embeds the residual semantic content for vector search. Example: _"diabetic patients on metformin who developed lactic acidosis"_ → filter: diagnosis ICD-10 E11.\*, medication RxNorm metformin; semantic: "developed lactic acidosis."

## 5.11 Evidence-Based Retrieval

All output is anchored to retrievable evidence: case passages, structured fields, and indexed medical literature (guidelines, PubMed abstracts loaded into a dedicated literature collection). Nothing is asserted without a pointer to its source.

## 5.12 Automatic Similarity Explanation

For each retrieved case, the explanation engine produces a structured rationale: shared symptoms, shared diagnoses (with codes), shared medications, comparable lab abnormalities, comparable demographics, comparable trajectories — each item cited. Generated via constrained, schema-validated LLM output (§13.7).

## 5.13 Case Comparison

Side-by-side comparison of the query patient against one or more retrieved patients: aligned timelines, entity diff (shared vs. unique conditions/medications), lab value comparison charts, and an LLM-generated comparative narrative describing where the cases converge and diverge.

## 5.14 Outcome Analysis

Across a retrieved cohort, the system aggregates documented outcomes by treatment path: "Among the 25 most similar cases, 15 received treatment A (12 documented improvement), 10 received treatment B (5 documented improvement)." Presented as descriptive retrieval statistics with explicit caveats — never as treatment recommendations. All numbers link back to the underlying cases.

## 5.15 Medical Literature Retrieval

A separate Qdrant collection indexes medical literature (clinical guidelines, open-access abstracts). When explaining a cohort, the system retrieves relevant literature passages and cites them alongside case evidence, clearly distinguishing "what our archive shows" from "what published literature says."

---

# 6. End-to-End System Behavior

## 6.1 Canonical Scenario

**Step 1 — Upload.** Dr. Reyes uploads a 6-page de-identified patient record: a scanned referral letter (PDF), a typed clinical note (DOCX), and a lab panel (CSV).

**Step 2 — Ingestion & AI processing (background).** The pipeline OCRs the scan, parses the note and labs, cleans the text, and extracts structured entities:

| Field           | Extracted                                                                            |
| --------------- | ------------------------------------------------------------------------------------ |
| Age             | 67                                                                                   |
| Gender          | Male                                                                                 |
| Symptoms        | exertional dyspnea, bilateral leg edema, orthopnea; _denies chest pain_ (negated)    |
| Diagnoses       | congestive heart failure (ICD-10 I50.9), type 2 diabetes (E11.9), hypertension (I10) |
| Medical history | CABG 2019, chronic kidney disease stage 3                                            |
| Medications     | furosemide 40 mg, metformin 1000 mg, lisinopril 20 mg                                |
| Lab results     | BNP 1250 pg/mL (high), creatinine 1.8 mg/dL (high), HbA1c 7.9% (high)                |
| Procedures      | echocardiogram (EF 35%)                                                              |
| Outcome         | (query patient — pending)                                                            |

**Step 3 — Embedding.** The record is chunked by clinical section, embedded with BGE-M3 (dense + sparse), and a composite case vector plus per-section vectors are computed.

**Step 4 — Retrieval.** Qdrant executes filtered hybrid search: dense similarity over case vectors, sparse similarity over lexical weights, payload filter (adult, male ± configurable, cardiac cohort optional), fused and reranked. Top-10 cases return with scores.

**Step 5 — Explanation.** For the top case (similarity 0.91, confidence "high"), the LLM produces:

> _"This case is highly similar: both patients are men in their late 60s presenting with exertional dyspnea and peripheral edema [Case 4821, note §2]; both carry diagnoses of CHF with reduced EF (35% vs 30%) [echo report], T2DM, and CKD stage 3 [problem list]; medication overlap includes loop diuretic and ACE inhibitor therapy [medication list]. In Case 4821, the care team additionally initiated an SGLT2 inhibitor; documented outcome was symptom improvement and discharge at day 6 without readmission within the recorded 90-day window [discharge summary]. Retrieved literature: 2 guideline passages on HFrEF management in CKD [Lit-118, Lit-204]. This is retrieved evidence, not a treatment recommendation."_

**Step 6 — Review & compare.** Dr. Reyes opens the case comparison view, inspects aligned timelines and lab trends, filters the cohort to cases with 90-day outcome data, and reviews the outcome analysis panel.

**Step 7 — Export.** She exports a cited PDF report of the retrieval session for the case file. Every step was audit-logged.

## 6.2 What the System Never Does

- It never outputs "Diagnosis: X" for the query patient.
- It never says "you should prescribe/administer/order."
- It never fabricates a case, statistic, or citation — output validation rejects uncited claims.
- It never returns results the requesting user lacks permission to see.

---

# 7. Core Modules

This section specifies all platform modules. Each module description covers purpose, responsibilities, key behaviors, data owned, and interactions.

## 7.1 Authentication

**Purpose.** Verify user identity and establish secure sessions for a healthcare-grade environment.

**Responsibilities.**

- Credential-based login with strong password policy (length, complexity, breach-list screening) and salted adaptive hashing (Argon2id).
- JWT-based access tokens (short-lived, ~15 min) with rotating refresh tokens (httpOnly, secure cookies); refresh-token reuse detection triggers session revocation.
- Multi-factor authentication (TOTP) — mandatory for admin roles, configurable for clinical roles.
- Optional enterprise SSO via OIDC/SAML for hospital identity providers (design-ready in v1, implemented when required).
- Brute-force protection: progressive delays, account lockout with admin unlock, per-IP and per-account rate limits (Redis-backed counters).
- Session inventory per user with remote revocation ("sign out everywhere").
- Password reset via time-limited, single-use signed tokens; email-based, never revealing account existence.

**Data owned.** Credential hashes, MFA secrets (encrypted at rest), refresh-token family records, login attempt history.

**Interactions.** Issues identity context consumed by User Management (roles) and Audit Logs (every auth event is logged: success, failure, lockout, MFA challenge, token refresh, logout).

## 7.2 User Management

**Purpose.** Manage user lifecycle, roles, and permissions (RBAC).

**Roles (v1).**

| Role           | Capabilities                                                                               |
| -------------- | ------------------------------------------------------------------------------------------ |
| `admin`        | Full platform administration: users, datasets, configuration, model settings, audit access |
| `physician`    | Search, view cases, compare, export reports, annotate results                              |
| `researcher`   | Search and analytics over de-identified cohorts; no re-identification views                |
| `data_manager` | Dataset upload, ingestion monitoring, document management; no clinical search              |
| `auditor`      | Read-only access to audit logs and compliance reports                                      |

**Responsibilities.**

- User CRUD with invitation-based onboarding (admin invites; user activates via emailed link).
- Role assignment and fine-grained permission checks enforced server-side on every request (deny-by-default).
- Organization/department scoping: users see only datasets their department is granted.
- Profile management: name, department, specialty, notification preferences, UI preferences.
- Deactivation (soft) and scheduled deletion honoring audit retention requirements.

**Interactions.** Consumes identity from Authentication; permission decisions are consulted by every API-facing module; all role changes are audit-logged with actor, target, before/after.

## 7.3 Dataset Management

**Purpose.** Organize patient records into governed datasets — the unit of ingestion, access control, and evaluation.

**Concepts.**

- **Dataset**: a named collection of patient records with shared provenance (e.g., "Cardiology Archive 2015–2022", "MIMIC-IV Demo Subset"), a schema mapping profile, a de-identification status, and an access policy.
- **Ingestion batch**: one upload session into a dataset, tracked end-to-end (files received → parsed → extracted → embedded → indexed) with per-document status and error detail.

**Responsibilities.**

- Dataset CRUD; per-dataset settings: language, expected document types, entity extraction profile, embedding model version, target Qdrant collection.
- Batch upload orchestration (files or archive bundles), duplicate detection (content hashing), and idempotent re-ingestion.
- Lifecycle states: `draft → ingesting → indexing → active → archived`; searches only touch `active` datasets.
- Versioned reindexing: when the embedding model or chunking policy changes, a dataset can be re-embedded into a new collection version and atomically switched, with rollback.
- Dataset statistics: record counts, entity distribution, code coverage (ICD/LOINC/RxNorm mapping rates), ingestion error rates.

**Interactions.** Feeds documents into the OCR/parsing pipeline; publishes indexing jobs to background workers; exposes dataset filters to Similarity Search; supplies corpora to the Evaluation Dashboard.

## 7.4 Patient Records

**Purpose.** Canonical structured representation of each patient case after AI processing — the "golden record" the rest of the system reasons over.

**Structure (logical).**

- **Patient case**: pseudonymous case ID, demographics (age at encounter, sex), dataset linkage, de-identification flags.
- **Encounters**: admissions/visits with dates (or normalized relative dates), department, disposition.
- **Clinical entities** (per encounter, all with source-span provenance): symptoms (with negation/severity), diagnoses (ICD-10 + SNOMED), medications (RxNorm, dose, route, status), lab results (LOINC, value, unit, flag, timestamp), procedures (code, date), outcomes (disposition, complications, readmission, mortality flags where documented).
- **Narrative sections**: cleaned text segments (chief complaint, HPI, assessment, plan, discharge summary) with offsets into the original document.
- **Provenance**: every structured fact links to (document ID, page, character span) so the UI can highlight the exact source text.

**Responsibilities.**

- Persist and version the structured record; re-processing creates a new version, never silently overwrites.
- Serve the Patient Viewer, Timeline, and Comparison views.
- Provide the payload fields synchronized into Qdrant for filtering.

**Interactions.** Written by the Medical NLP Pipeline; read by Search, Explanation Engine (context construction), Timeline, Analytics.

## 7.5 Medical Document Upload

**Purpose.** Safe, resumable intake of clinical files of all supported types (§9).

**Responsibilities.**

- Chunked, resumable uploads for large files; drag-and-drop and folder/archive upload in the UI.
- Strict validation: file-type sniffing (magic bytes, not extensions), size limits, archive-bomb protection, malware scanning hook, rejection of executables and active content.
- PDF sanitization (strip JavaScript/embedded files) before storage.
- Content-addressed storage (SHA-256) in the object store; duplicates detected and linked rather than re-stored.
- Upload manifests: which files belong to which case/batch, declared document types, and optional metadata sidecars.
- Quarantine state for files failing validation, with admin review queue.

**Interactions.** Hands validated files to the OCR Pipeline / Document Parsing dispatcher via the job queue; reports per-file status to Dataset Management.

## 7.6 OCR Pipeline

**Purpose.** Convert scanned/image documents into machine-readable text with layout awareness.

**Engine.** PaddleOCR (PP-OCR + PP-Structure), self-hosted, GPU-optional.

**Responsibilities.**

- Pre-processing: page rasterization, deskewing, denoising, contrast normalization, orientation detection, DPI upscaling for low-quality faxes.
- Detection & recognition: text-line detection, recognition with per-line confidence scores; language model selection per dataset language.
- Layout analysis: distinguish paragraphs, tables (critical for lab reports), headers/footers, stamps, and handwriting regions (flagged, not transcribed in v1).
- Table reconstruction: lab tables recovered as structured rows (analyte, value, unit, reference range) rather than jumbled text.
- Quality gating: pages with mean confidence below threshold are flagged `low_ocr_quality`; downstream extraction records this so confidence propagates to search results.
- Output: normalized text with page/region/bounding-box provenance retained for UI highlighting.

**Interactions.** Triggered for scanned PDFs and images; output flows into Document Parsing; OCR confidence metadata flows all the way to the Patient Viewer.

## 7.7 Document Parsing

**Purpose.** Turn heterogeneous machine-readable inputs into a single normalized internal document model.

**Responsibilities per format.**

- **Text PDF**: native text extraction with layout ordering; fallback to OCR when the text layer is missing/garbage (heuristic detection).
- **DOCX**: structure-aware extraction (headings, lists, tables).
- **Plain text / medical notes**: encoding detection, normalization.
- **FHIR JSON (R4)**: resource-level mapping — `Patient`, `Condition`, `MedicationRequest`/`MedicationStatement`, `Observation`, `Procedure`, `Encounter`, `DiagnosticReport`, `DocumentReference` — directly into the structured record, bypassing NLP for already-coded fields (NLP still runs on narrative `text` elements).
- **HL7 v2**: segment parsing (PID, DG1, OBX, OBR, RXA/RXE, PV1) with configurable mapping profiles per sending system.
- **CSV**: per-dataset column-mapping profiles (declared at dataset setup) for tabular labs/med lists; validation with row-level error reporting.
- **Radiology/laboratory reports**: section-aware parsing (indication, technique, findings, impression / panel tables).

**Common responsibilities.**

- Sectionizer: detect canonical clinical sections (chief complaint, HPI, PMH, medications, allergies, labs, assessment & plan, discharge summary) using header patterns and ML fallback.
- Emit the **Normalized Document Model**: ordered sections → blocks (paragraph/table/list) → text with provenance (source file, page, offsets).

**Interactions.** Consumes OCR output and native files; produces input for the Medical NLP Pipeline; parsing failures are surfaced per-document in Dataset Management.

## 7.8 Medical NLP Pipeline

**Purpose.** Extract, qualify, and normalize clinical meaning from text. This is the heart of "understanding" (§3.2 G1–G7). Full linguistic detail in §15.

**Stages.**

1. **De-identification (safety stage)**: detect and mask residual PHI (names, MRNs, dates→shifted, phones, addresses) before anything is indexed; configurable strictness per dataset.
2. **Clinical NER**: symptoms, diseases/diagnoses, medications, dosages, lab analytes and values, procedures, anatomical sites, temporal expressions.
3. **Assertion & context classification**: negation ("denies chest pain"), uncertainty ("possible PE"), subject (patient vs. family history), temporality (current vs. historical), conditionality.
4. **Entity normalization**: map mentions to ICD-10, SNOMED CT, RxNorm, LOINC (see §15); abbreviation expansion with context disambiguation.
5. **Relation extraction**: drug–dosage, drug–indication, lab–value–time, problem–procedure linkages.
6. **Temporal anchoring**: normalize temporal expressions and order events for the timeline (§15.11).
7. **Section-aware summarization fields**: derive the canonical facet texts (symptom summary, diagnosis list, medication list, lab abnormality summary, history summary, outcome summary) used by the embedding pipeline.

**Quality behavior.** Every extraction carries a confidence score and provenance span; low-confidence extractions are stored but flagged and excluded from strict filters by default.

**Interactions.** Writes the structured Patient Record (§7.4); feeds facet texts to Embedding Generation; extraction quality metrics stream to the Evaluation Dashboard.

## 7.9 Embedding Generation

**Purpose.** Convert normalized clinical text into vector representations. (Pipeline detail in §12.)

**Responsibilities.**

- Section-aware chunking, cleaning, and preparation of embedding inputs.
- Batch inference with BGE-M3 producing dense vectors (1024-d), sparse lexical weight vectors, and optionally multi-vector outputs for reranking experiments.
- Composite case-vector computation (weighted aggregation of facet vectors) plus per-facet vectors.
- Model/version stamping on every vector; deterministic re-embedding for reindexing.
- Embedding cache (Redis + content hash) to skip unchanged text on re-processing.
- Throughput management: GPU batch scheduling, backpressure to the job queue.

**Interactions.** Consumes NLP output; writes vectors + payloads to Qdrant; registers vector↔record linkage in PostgreSQL.

## 7.10 Vector Database (Qdrant Service Module)

**Purpose.** Own all interaction with Qdrant: collections, upserts, searches, filters, lifecycle. (Design detail in §11.)

**Responsibilities.**

- Collection management with versioned naming (`cases_v3_bgem3`), snapshot backups, and blue/green collection switching for reindexing.
- Upsert/delete operations with idempotency; consistency reconciliation job comparing PostgreSQL record registry vs. Qdrant points.
- Query execution: dense, sparse, hybrid, filtered, grouped (by patient) searches; recommendation-style "more like this case" queries.
- Health, capacity, and index-parameter monitoring (HNSW `m`, `ef_construct`, `ef_search` tuning surface).

**Interactions.** Called by Embedding Generation (writes) and Similarity Search/Hybrid Retrieval (reads); metrics into Monitoring.

## 7.11 Metadata Database (PostgreSQL Service Module)

**Purpose.** Own all structured, transactional data (design detail in §10.1): users, datasets, documents, patient records, entities, jobs, annotations, audit anchors, evaluation data.

**Responsibilities.**

- Schema migrations (versioned, reversible), referential integrity, transactional writes for multi-step ingestion bookkeeping.
- Query services for filters, analytics aggregations, and the joins that decorate vector-search results with full record data.
- Row-level access filtering aligned with dataset permissions.

## 7.12 Similarity Search

**Purpose.** The core product capability: given a query (patient record or natural-language text), return the most clinically similar cases.

**Responsibilities.**

- Query construction: embed query text/case facets; parse natural-language constraints into filters (§5.10); validate user-selected filters.
- Execute retrieval via the Hybrid Retrieval module; apply permission filtering; assemble result envelopes: case summary card, overall score, facet score breakdown, confidence label, matched entities.
- Search sessions: persist query, filters, results, and scores so sessions are reviewable, comparable, exportable, and auditable.
- "More like this" pivots from any retrieved case.
- Saved searches and re-run with change detection ("2 new similar cases since last run").

**Interactions.** Orchestrates Hybrid Retrieval and the LLM Explanation Engine (explanations are generated lazily per viewed result to control cost); logs to Audit; emits quality signals (user clicks, accepts, dismisses) to Evaluation.

## 7.13 Hybrid Retrieval

**Purpose.** Implement the multi-stage retrieval strategy (full detail in §14): dense + sparse candidate generation → fusion → metadata filtering → reranking → top-K selection.

**Responsibilities.**

- Parallel dense and sparse queries against Qdrant with shared filters.
- Reciprocal Rank Fusion (and configurable weighted-score fusion) of candidate lists.
- Cross-encoder reranking of the fused top-N (N≈100) down to top-K (K≈10–25).
- Score calibration and confidence derivation (margins, absolute thresholds).
- Retrieval configuration profiles (fast / balanced / thorough) selectable per query; all parameters logged for reproducibility.

## 7.14 LLM Explanation Engine

**Purpose.** Generate grounded, cited, structured explanations and comparative narratives. (Full detail in §13.)

**Responsibilities.**

- Context construction from retrieved passages, structured entity summaries, and literature snippets within token budgets.
- Prompt template management (versioned, tested): similarity explanation, case comparison, cohort outcome summary, timeline narrative.
- Structured JSON output with schema validation; citation resolution (every claim → source span).
- Groundedness verification pass; refusal behaviors for diagnostic requests; safety disclaimer injection.
- Response caching (Redis) keyed on (template version, context hash); cost and latency accounting per generation.

## 7.15 Medical Timeline

**Purpose.** Transform extracted temporal events into an interactive chronological representation.

**Responsibilities.**

- Event model: typed events (symptom onset, diagnosis, medication start/stop, lab result, procedure, admission/discharge, outcome) with normalized timestamps or relative anchors and confidence.
- Conflict handling: contradictory or unanchorable dates flagged, not guessed.
- Timeline API for the viewer: zoomable ranges, event filtering by type, lab-trend series extraction.
- Dual-timeline alignment for Case Comparison (align on admission date or index event).

## 7.16 Analytics Dashboard

**Purpose.** Operational and clinical-archive insight for authorized users.

**Content.**

- Archive analytics: dataset sizes, diagnosis distribution (ICD chapters), medication frequency, lab abnormality prevalence, demographic distributions, outcome class distributions.
- Usage analytics: searches per day, active users, most-queried conditions, explanation view rates, export counts.
- Cohort analytics: for a retrieved cohort — treatment-path breakdowns, outcome distributions, length-of-stay distributions (descriptive only, with sample-size caveats displayed).
- All charts rendered with Apache ECharts; every aggregate links to its underlying (permitted) records.

## 7.17 Evaluation Dashboard

**Purpose.** Continuous, visible measurement of AI quality — the feature that separates this platform from a demo.

**Content.**

- **Retrieval metrics**: Recall@K, Precision@K, MRR, nDCG on curated gold query sets; dense vs. sparse vs. hybrid vs. reranked ablation views; trend lines across model/config versions.
- **Extraction metrics**: NER precision/recall/F1 per entity type against annotated samples; normalization (code-mapping) accuracy; negation detection accuracy.
- **Generation metrics**: groundedness score (fraction of claims attributable to context), citation validity rate, refusal correctness on adversarial diagnostic prompts, human review ratings.
- **Operational metrics**: latency percentiles per pipeline stage, token/cost per explanation, cache hit rates, OCR confidence distributions.
- **Feedback loop**: physician accept/dismiss/flag signals aggregated into weak relevance labels; annotation queue for building gold sets.

## 7.18 Administration Panel

**Purpose.** Single control surface for platform operators.

**Content.**

- User & role administration; dataset permission grants; invitation management.
- Pipeline configuration: OCR settings, NLP profiles, embedding model versions, retrieval profiles, prompt template activation, safety-filter thresholds.
- Job control: queue depths, retry/requeue failed jobs, dead-letter inspection.
- System settings: retention policies, export policies, rate limits, feature flags.
- Environment health summary and links into Monitoring dashboards.

## 7.19 Audit Logs

**Purpose.** Immutable, queryable trail of every security- and clinically-relevant action — a compliance requirement, not a nice-to-have.

**Logged events.** Authentication events; permission changes; dataset lifecycle actions; document uploads/downloads/deletions; every search (query text, filters, result IDs, scores); every record view; every LLM generation (template version, context hash, output hash); every export; admin configuration changes.

**Properties.**

- Append-only storage with hash chaining per shard for tamper evidence; write path is asynchronous but loss-intolerant (queue with persistence).
- Structured entries: actor, role, action, resource, timestamp, IP/session, outcome, correlation ID linking a user action across services.
- Retention per policy (default 6 years, configurable); auditor role gets a dedicated search UI with export.
- Audit access is itself audited.

## 7.20 Monitoring

**Purpose.** Observability across services and the AI pipeline.

**Scope.**

- **Metrics** (Prometheus-compatible): request rates/latency/errors per endpoint; queue depths and job durations per stage; GPU/CPU/memory; Qdrant query latency and collection sizes; LLM latency, token usage, error/refusal rates; cache hit ratios.
- **Logs**: structured JSON logs, centralized, correlation-ID threaded through API → worker → LLM call.
- **Traces**: distributed tracing across the search path (API → retrieval → rerank → LLM) to localize latency.
- **Alerts**: error-rate spikes, queue backlog thresholds, OCR failure surges, groundedness-score regression, disk/GPU saturation, certificate expiry.
- **Dashboards**: Grafana boards per domain (API, ingestion, retrieval, LLM, infrastructure).

---

# 8. The AI Pipeline

The pipeline is a directed flow of eleven stages. Each stage is independently deployable as a worker, idempotent, resumable, and emits metrics. A document's progress through the stages is tracked as a state machine in PostgreSQL.

```
Raw document → OCR → Cleaning → Medical entity extraction → Normalization
→ Embedding generation → Vector database → Similarity search → Reranking
→ Context construction → LLM reasoning → Explanation generation → Response
```

## 8.1 Stage 1 — Raw Document Intake

**Input:** uploaded file (any supported type). **Output:** validated, content-addressed stored object + ingestion job.

Validation (type sniffing, size, sanitization, malware hook), deduplication by SHA-256, manifest linkage to dataset/case, and job enqueue. Failures land in quarantine with human-readable reasons. Nothing unvalidated ever reaches later stages.

## 8.2 Stage 2 — OCR

**Input:** scanned PDFs/images (skipped for born-digital text). **Output:** text with layout + per-region confidence.

PaddleOCR performs preprocessing (deskew, denoise, orientation), detection, recognition, and PP-Structure layout analysis with table reconstruction. Per-page confidence gates: high-confidence pages proceed; low-confidence pages proceed _flagged_, and the flag propagates to every downstream artifact derived from them. Bounding-box provenance is preserved so the UI can highlight source pixels.

## 8.3 Stage 3 — Cleaning

**Input:** raw extracted text. **Output:** normalized text in the document model.

- Unicode normalization (NFKC), whitespace/hyphenation repair (OCR line-break healing), header/footer and page-number stripping, boilerplate removal (fax banners, "CONFIDENTIAL" stamps).
- De-hyphenation and sentence-boundary restoration tuned for clinical text (abbreviation-aware: "Dr.", "b.i.d.", "q.d." must not split sentences).
- Encoding repair for legacy exports; control-character stripping.
- Sectionization into the canonical clinical sections (§7.7).
- **De-identification pass** (Philter-style rules + NER): residual names, identifiers, contact details masked; dates consistently shifted per-patient to preserve intervals. This stage is mandatory before any text is embedded or indexed.

## 8.4 Stage 4 — Medical Entity Extraction

**Input:** clean sectioned text. **Output:** typed, qualified entity mentions with spans and confidence.

Clinical NER (§15) extracts symptoms, diagnoses, medications + attributes, labs + values, procedures, anatomy, and temporal expressions. Assertion classification qualifies each mention (negated / uncertain / historical / family). Relation extraction links attributes (drug↔dose, lab↔value↔time). Every mention keeps (section, char span, page) provenance.

## 8.5 Stage 5 — Normalization

**Input:** entity mentions. **Output:** coded, canonical entities.

Mentions map to standard vocabularies — diagnoses → ICD-10 + SNOMED CT; medications → RxNorm ingredient level; labs → LOINC with unit normalization (UCUM) and reference-range flagging; procedures → SNOMED/ICD-10-PCS. Abbreviations are expanded with context disambiguation ("PE": pulmonary embolism vs. physical exam, resolved by section and surrounding entities). Unmappable mentions are retained as free-text entities with a `unmapped` flag and surfaced in dataset quality stats.

## 8.6 Stage 6 — Embedding Generation

**Input:** facet texts + narrative chunks. **Output:** dense + sparse vectors, versioned.

Section-aware chunking (§12.1), BGE-M3 batch inference producing dense (1024-d) and sparse lexical vectors, composite case-vector aggregation, cache lookup by content hash, and version stamping. GPU workers pull batches from the queue; backpressure protects the GPU from overload.

## 8.7 Stage 7 — Vector Database Indexing

**Input:** vectors + payloads. **Output:** searchable points in Qdrant.

Idempotent upserts keyed by deterministic point IDs (`case:{id}:facet:{name}:v{n}`), payload synchronization (demographics, codes, dataset, dates, outcome class, quality flags), and registry bookkeeping in PostgreSQL. A reconciliation job continuously verifies PostgreSQL↔Qdrant consistency.

## 8.8 Stage 8 — Similarity Search

**Input:** user query (case or text) + filters. **Output:** candidate set (~100–200).

Query embedding (dense + sparse), natural-language constraint parsing into payload filters, parallel filtered dense and sparse ANN queries, candidate union. Permission filters are always injected server-side.

## 8.9 Stage 9 — Reranking

**Input:** fused candidates. **Output:** precision-ordered top-K with calibrated scores.

Reciprocal Rank Fusion merges dense/sparse lists; a cross-encoder reranker (BGE-reranker class) scores (query, candidate) pairs over richer text; facet-score decomposition is computed; confidence labels derived from score distribution and margins. Rerank depth and model are per-profile configuration.

## 8.10 Stage 10 — Context Construction

**Input:** top-K cases + query record. **Output:** token-budgeted, structured LLM context.

Selection of the most relevant passages per case (entity-dense, query-relevant sections first), structured entity summaries (compact tabular text), literature retrieval from the literature collection, source labeling (`[Case 4821 §Assessment]`, `[Lit-118]`), deduplication, and token-budget packing with priority ordering. The context is hashed for caching and audit.

## 8.11 Stage 11 — LLM Reasoning & Explanation Generation

**Input:** context + versioned prompt template. **Output:** validated, structured explanation.

Gemini generates schema-constrained JSON (rationale items, each with citation IDs; facet commentary; outcome summary; confidence statement; mandatory disclaimer). Post-generation validation: schema check, citation resolution (every claim's citation must exist in context), groundedness scoring, banned-phrase screen (diagnostic/prescriptive language). Failures trigger one constrained retry, then graceful degradation to non-LLM structured explanation (scores + matched entities only).

## 8.12 Stage 12 — Response

**Input:** validated explanation + result envelope. **Output:** API response / UI rendering.

Assembly of the final payload: ranked cases, scores, facet breakdowns, confidence, explanations with resolvable citations, cohort outcome statistics, and audit record. Responses are cached (short TTL) keyed by (query hash, filters, config version).

**Cross-cutting pipeline guarantees:** every stage is idempotent and retry-safe; every artifact is versioned; every stage emits duration/success metrics; a document can be replayed from any stage; and provenance survives end-to-end from pixel to citation.

---

# 9. Supported Document Types

| Type                   | Ingestion path                                      | Notes                                                                                   |
| ---------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **PDF (text layer)**   | Native extraction → cleaning                        | Fallback to OCR if text layer is absent/corrupt (heuristic: extractable chars per page) |
| **Scanned PDF**        | Rasterize → PaddleOCR → layout/table reconstruction | Per-page confidence gating; handwriting regions flagged                                 |
| **Word (DOCX)**        | Structure-aware parsing                             | Headings/tables preserved into the document model                                       |
| **Plain text (TXT)**   | Encoding detection → cleaning                       | Common for exported notes                                                               |
| **FHIR JSON (R4)**     | Resource mapping                                    | Coded fields bypass NLP; narrative elements still processed                             |
| **HL7 v2 messages**    | Segment parsing with mapping profiles               | PID/DG1/OBX/OBR/RXE/PV1; per-source profiles                                            |
| **CSV**                | Column-mapping profiles per dataset                 | Labs, med lists, encounter tables; row-level validation                                 |
| **Medical notes**      | Text/PDF/DOCX paths + sectionizer                   | Progress notes, consults, discharge summaries                                           |
| **Laboratory reports** | Table-centric parsing (native or OCR)               | Analyte/value/unit/range extraction → LOINC                                             |
| **Radiology reports**  | Section-aware parsing                               | Indication/technique/findings/impression; impression weighted higher in embeddings      |

Unsupported or unrecognized files are rejected at intake with explicit reasons; partially parseable files produce partial records flagged `incomplete` rather than silent data loss.

---

# 10. Database Design

The platform uses three specialized stores. The rule: **PostgreSQL is the source of truth for facts, Qdrant is the source of truth for similarity, Redis holds nothing that cannot be regenerated.**

## 10.1 PostgreSQL — Structured Source of Truth

PostgreSQL stores all transactional, relational, and auditable data:

- **Identity & access**: users, roles, permissions, sessions/refresh-token families, MFA enrollment, invitations.
- **Datasets & ingestion**: datasets, ingestion batches, documents (with content hashes, storage keys, type, status), per-stage pipeline state and error records.
- **Patient records (structured)**: cases, encounters, and all normalized clinical entities — diagnoses (ICD-10/SNOMED), medications (RxNorm + attributes), lab results (LOINC, value, unit, flag, time), procedures, symptoms (with assertion status), outcomes — each with provenance spans and confidence. Narrative sections stored with offsets.
- **Vector registry**: mapping of every Qdrant point ID → (case, facet, chunk, embedding model version) for consistency reconciliation and safe deletion.
- **Search & session data**: search sessions, queries, applied filters, returned result sets with scores, user feedback (accept/dismiss/flag), saved searches, exports.
- **LLM bookkeeping**: prompt template versions, generation records (template version, context hash, output hash, token counts, latency, validation outcome) — content-addressed, enabling exact reproduction of what the model saw.
- **Evaluation data**: gold query sets, relevance judgments, annotation tasks, metric snapshots per configuration version.
- **Audit anchors**: append-only audit tables (hash-chained), admin configuration history.
- **Terminology cache**: local subsets of ICD-10, SNOMED CT, LOINC, RxNorm concept tables used by normalization (licensed content handled per deployment).

Characteristics: strict migrations, foreign-key integrity, row-level dataset scoping, partitioning for high-volume tables (lab results, audit logs), and read-replicas for analytics if needed.

## 10.2 Qdrant — Vector Search Engine

Qdrant stores everything needed to answer "what is similar" without a round-trip to PostgreSQL (full design in §11):

- **Dense vectors** per case (composite) and per facet/chunk (BGE-M3, 1024-d, cosine).
- **Sparse vectors** (BGE-M3 lexical weights) for the hybrid path.
- **Payloads**: pseudonymous case ID, dataset ID, facet type, age band, sex, encounter year, department, ICD-10 code list, medication (RxNorm) list, outcome class, quality flags, embedding version.
- **Collections**: `cases_v{n}` (patient facets/chunks), `literature_v{n}` (guideline/abstract passages).

Qdrant is treated as **rebuildable**: it can be fully reconstructed from PostgreSQL + object storage at any time (this is how reindexing works). Snapshots provide fast recovery; PostgreSQL provides ultimate recovery.

## 10.3 Redis — Cache, Queues, and Ephemeral State

Redis holds only regenerable, expiring data:

- **Caches**: embedding cache (content hash → vector, for re-ingestion speedups), search-result cache (query hash → response, short TTL), LLM explanation cache (context hash + template version → validated output), terminology lookup cache, rendered analytics aggregates.
- **Queues & coordination**: background job queues per pipeline stage (with persistence enabled), rate-limit counters (per-user, per-IP), distributed locks for reindex/reconciliation jobs, WebSocket/session presence for live ingestion progress.
- **Session support**: short-lived token blacklist (revoked JWTs until expiry), login attempt counters.

Eviction policy is deliberate: caches use LRU with TTLs; queues and rate-limit state are non-evictable. Loss of Redis degrades performance and delays jobs but never loses facts.

## 10.4 Object Storage

Original uploaded files, OCR page images, and generated export PDFs live in S3-compatible object storage (MinIO self-hosted / S3 in cloud), content-addressed, encrypted at rest, referenced by storage keys from PostgreSQL. Nothing binary is stored in the databases.

---

# 11. Vector Database Design

## 11.1 Collections

| Collection        | Contents                    | Point granularity                                                                    |
| ----------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| `cases_v{n}`      | Patient case vectors        | One point per (case × facet) + one composite point per case + narrative chunk points |
| `literature_v{n}` | Medical literature passages | One point per passage (guideline paragraph / abstract)                               |

Versioned collection names support blue/green reindexing: a new embedding model or chunking policy writes `cases_v4`, evaluation compares it against `cases_v3`, and an atomic alias switch promotes it. Rollback is instant.

**Facet vectors per case:** `composite` (whole-case), `symptoms`, `diagnoses`, `medications`, `labs`, `history`, `outcome`, plus `narrative:{chunk_i}` for long note passages. Facet points share the case's payload and allow per-facet similarity scoring and facet-restricted search ("find cases with similar lab patterns regardless of diagnosis").

## 11.2 Embeddings

- **Model**: BGE-M3; dense 1024-dimensional vectors; long-context (up to 8192 tokens) so whole sections embed without aggressive truncation.
- **Named vectors**: each point carries a named dense vector (`dense`) and a named sparse vector (`sparse`) — Qdrant's multi-vector points keep both retrieval modes on the same point with the same payload.
- **Normalization**: vectors L2-normalized at write time; cosine similarity thereby equals dot product.
- **Version stamping**: `embedding_version` in the payload; mixed-version collections are forbidden (enforced by the indexing service).

## 11.3 Metadata (Payload) Schema

Every point carries a filterable payload:

| Field               | Type                                                         | Purpose                                            |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| `case_id`           | keyword                                                      | Grouping, joins back to PostgreSQL                 |
| `dataset_id`        | keyword                                                      | Access control, dataset-scoped search              |
| `facet`             | keyword                                                      | Facet-restricted search                            |
| `age_band`          | keyword (e.g., `60-69`)                                      | Demographic filtering without exact-age PHI        |
| `sex`               | keyword                                                      | Filtering                                          |
| `encounter_year`    | integer                                                      | Temporal filtering                                 |
| `department`        | keyword                                                      | Cohort scoping                                     |
| `icd10_codes`       | keyword[]                                                    | Diagnosis filtering (prefix-match capable: `E11*`) |
| `rxnorm_ids`        | keyword[]                                                    | Medication filtering                               |
| `loinc_abnormal`    | keyword[]                                                    | Abnormal-lab filtering                             |
| `outcome_class`     | keyword (`improved/unchanged/deteriorated/deceased/unknown`) | Outcome filtering                                  |
| `quality_flags`     | keyword[] (`low_ocr_quality`, `incomplete`)                  | Quality-aware filtering                            |
| `embedding_version` | keyword                                                      | Reindex integrity                                  |

Payload indexes are created for every filterable field. Payloads deliberately exclude free text and any direct identifiers.

## 11.4 Similarity Metrics

- **Dense**: cosine similarity (on normalized vectors). Chosen over Euclidean because clinical text embeddings care about direction (semantic orientation), not magnitude, and BGE-M3 is trained for cosine retrieval.
- **Sparse**: dot product over sparse lexical weight vectors (BM25-like behavior with learned term weights).
- **Index**: HNSW for dense (tunable `m`, `ef_construct` at build; `ef_search` per query profile — fast/balanced/thorough); inverted index for sparse.
- Score ranges are normalized to [0,1] at the service layer so downstream fusion, calibration, and UI display are metric-agnostic.

## 11.5 Hybrid Search

Executed as parallel dense and sparse queries with identical filters, fused via Reciprocal Rank Fusion (rank-based, robust to score-scale mismatch) with a configurable weighted-score alternative. Rationale in §14.3. Qdrant's Query API hybrid support is used where it simplifies orchestration; the fusion logic remains observable and configurable at the service layer for evaluation ablations.

## 11.6 Filtering

- **Pre-filtering inside ANN**: filters are passed into the Qdrant query so HNSW traversal respects them — no wasteful over-fetching and post-filtering. This is the mechanism behind Metadata Filtering (§5.5).
- **Mandatory injected filters**: `dataset_id ∈ user's permitted datasets` and `embedding_version = active` are added server-side on every query and cannot be overridden by clients.
- **Filter grammar exposed to users**: equality, set membership, numeric/date ranges, code-prefix matching, quality-flag exclusion, and boolean combinations (AND/OR/NOT) — surfaced through both the filter UI and natural-language query parsing.
- **Group-by-case**: facet/chunk points are grouped so one case never floods the result list; the best-scoring point represents the case and per-facet scores are reported alongside.

---

# 12. Embedding Pipeline

## 12.1 Chunking

Chunking is **section-aware and semantically bounded**, never naive fixed-window:

- **Facet texts** (symptom summary, diagnosis list, medication list, lab abnormality summary, history summary, outcome summary) are compact by construction (produced by the NLP pipeline) and embed as single units.
- **Narrative sections** (HPI, assessment & plan, discharge summary) are chunked at paragraph/sentence-group boundaries targeting ~350–512 tokens with ~15% overlap; a chunk never crosses a section boundary.
- **Tables** (labs) are linearized row-wise into "analyte: value unit (flag)" text before embedding — tables are never embedded as raw grids.
- Every chunk records (document, section, char span) provenance and a stable chunk ID so re-chunking is diffable.
- Chunking policy is versioned; changing it triggers dataset re-embedding into a new collection version.

## 12.2 Cleaning (Embedding-Specific)

Beyond pipeline Stage 3 cleaning, embedding inputs get: template boilerplate removal (EHR headers repeated per page), de-identification placeholder normalization (`[NAME]`, `[DATE:+3d]` → neutral tokens so placeholders don't dominate similarity), lowercasing left to the model (BGE-M3 is cased), and length guards (truncation warnings logged, not silent).

## 12.3 Medical Tokenization

BGE-M3's tokenizer handles general text; the pipeline compensates for clinical quirks _before_ tokenization: abbreviation expansion (post-disambiguation: "c/o SOB" → "complains of shortness of breath"), unit normalization ("mg/dL" standardized), dosage spacing repair ("40mg"→"40 mg"), and preservation of code tokens (ICD/LOINC codes kept verbatim — they carry exact-match value in the sparse channel). The expanded, normalized text is what gets embedded; the original text is what gets displayed.

## 12.4 Embedding Models

- **Primary**: **BGE-M3** — one model, three outputs (dense 1024-d, sparse lexical weights, multi-vector). Chosen for: long context (8192 tokens), multilingual coverage (hospital archives are rarely monolingual), strong retrieval benchmarks, and built-in hybrid capability that removes the need for a separate BM25 stack.
- **Reranker**: BGE-reranker-v2 class cross-encoder for Stage 9 (§14.4).
- **Evaluation harness** treats the embedding model as a swappable component: candidate models (e.g., domain-adapted clinical embedders) are benchmarked on the gold retrieval set before any promotion; promotion = new collection version, never in-place mutation.

## 12.5 Storage

Vectors live only in Qdrant (with named dense+sparse per point, payload attached). PostgreSQL stores the vector registry (point ID ↔ record ↔ model version), not vectors. The embedding cache in Redis stores (content hash + model version) → vector for re-ingestion acceleration.

## 12.6 Indexing

Batch upserts (256–1024 points) with idempotent deterministic IDs; HNSW build parameters tuned per collection size; payload indexes created before bulk load completes; indexing throughput and lag (documents processed vs. indexed) exported to Monitoring. During initial bulk ingestion, HNSW indexing can be deferred (index-after-load) for throughput.

## 12.7 Caching

Three cache layers: (1) embedding cache (skip recompute for identical text — very common on re-ingestion and dataset copies), (2) query-embedding cache (repeated/saved searches), (3) no caching of _stale_ similarity results beyond short TTL — correctness beats speed here. All keys include model version.

## 12.8 Updating

Record edits or re-processing produce a new record version → affected facets/chunks re-embedded → upsert by deterministic ID (overwrite) → registry version bump. Partial updates are supported (only changed facets re-embed). A nightly reconciliation job detects and heals PostgreSQL↔Qdrant drift.

## 12.9 Deletion

Deletion is a first-class flow (privacy requirement): deleting a case removes its Qdrant points (by `case_id` filter), its registry rows, its structured record (soft-delete then purge per retention policy), and its cached artifacts (embedding cache entries by content hash, result caches invalidated). Deletion completion is verified by the reconciliation job and audit-logged. Dataset deletion is a bulk version of the same flow with an explicit two-step confirmation.

---

# 13. LLM Module

## 13.1 Role and Boundaries

The LLM (Gemini) is used **exclusively as a grounded reasoning and explanation engine over retrieved context**. It is never a knowledge source, never a diagnostician, and never answers without context. Model calls go through a single internal LLM gateway service that enforces templates, budgets, validation, and logging.

## 13.2 Prompt Templates

Versioned, code-reviewed, and regression-tested like source code. Core templates:

| Template                      | Purpose                                                                     |
| ----------------------------- | --------------------------------------------------------------------------- |
| `similarity_explanation_v{n}` | Why is retrieved case X similar to the query case — factor by factor, cited |
| `case_comparison_v{n}`        | Structured comparative narrative for side-by-side view                      |
| `cohort_outcome_summary_v{n}` | Descriptive treatment/outcome aggregation across top-K, cited, caveated     |
| `timeline_narrative_v{n}`     | Short chronological narrative from extracted events                         |
| `query_understanding_v{n}`    | Natural-language query → structured filters + semantic residual             |
| `literature_link_v{n}`        | Connect cohort findings to retrieved literature passages                    |

Every template contains: role definition (clinical evidence assistant, non-diagnostic), hard behavioral rules (cite everything; say "insufficient evidence" when true; never diagnose/prescribe; never mention uncited facts), the structured context, the output JSON schema, and few-shot exemplars including a refusal exemplar.

## 13.3 Context Generation

Deterministic, budgeted assembly (§8.10): query-case structured summary → per-candidate best passages (entity-dense, query-relevant first) → per-candidate structured entity tables → literature passages → all wrapped with source tags (`[Case 4821 §A&P]`, `[Lit-118]`). Token budget is partitioned (e.g., 15% query case, 60% candidates, 15% literature, 10% instructions); overflow trims lowest-relevance passages first, never truncates mid-passage. The exact context is content-hashed and stored, making every generation reproducible.

## 13.4 Grounding

Grounding is enforced, not requested: (a) prompts forbid external knowledge assertions; (b) every output claim must carry a citation ID that exists in the supplied context; (c) a post-generation groundedness check verifies claim↔source entailment (NLI-style scoring, sampled human audit); (d) claims failing attribution are stripped or the generation is retried; (e) systematic groundedness scores feed the Evaluation Dashboard with regression alerts.

## 13.5 Hallucination Prevention

Layered defenses: retrieval-only knowledge policy; low temperature for factual templates; structured output schema (free prose minimized); citation-required validation; banned-phrase screening (diagnostic/prescriptive constructions like "the patient has", "should be treated with" targeting the _query_ patient); numeric-claim verification (any count/statistic must match the cohort statistics computed in code — the LLM is never the source of numbers, it narrates numbers computed deterministically); and graceful degradation — if validation fails twice, the UI shows the non-LLM structured explanation (scores, matched entities, outcome table) rather than risky prose.

## 13.6 Citations

Citation IDs resolve to (case, document, section, char span) or literature passage IDs. The frontend renders citations as clickable superscripts that open the source with the span highlighted. Citation validity (resolvable + entailing) is a tracked metric. Uncited sentences in factual sections are rejected by the validator.

## 13.7 Reasoning

For comparison and cohort templates, the model is instructed to reason stepwise _internally_ but emit only structured conclusions (rationale fields), keeping outputs compact and auditable. Comparative logic follows a fixed rubric: demographics → presentation → comorbidities → medications → labs → interventions → outcomes, so explanations are consistent and scannable across results.

## 13.8 Structured Output

All LLM responses are schema-validated JSON. Example (similarity explanation) — conceptual shape, not implementation:

- `similarity_factors[]`: `{facet, statement, citations[], strength}`
- `differences[]`: `{facet, statement, citations[]}`
- `treatments_observed[]`: `{treatment, outcome_note, citations[]}`
- `literature[]`: `{statement, citations[]}`
- `confidence`: `{level, basis}`
- `disclaimer`: fixed non-diagnostic disclaimer (validated present verbatim)

Schema violations trigger constrained regeneration once, then degradation (§13.5). Token usage, latency, template version, context hash, and validation outcome are logged per call; explanation cache (Redis) short-circuits repeat generations.

---

# 14. Retrieval Pipeline

## 14.1 Dense Retrieval

Query text/facets → BGE-M3 dense vector → filtered HNSW ANN search in Qdrant (cosine) → top-N per facet. Strengths: synonyms, paraphrase, abbreviation robustness, cross-lingual matching. Weakness: can blur exact tokens that matter (specific drug, specific mutation) — which is why it never runs alone.

## 14.2 Sparse Retrieval

Query → BGE-M3 sparse lexical weights → sparse index search (dot product) → top-N. Strengths: exact and rare-term precision (drug names, codes, rare diseases), interpretability (matched terms are visible). Learned sparse weights outperform plain BM25 by weighting clinically informative terms above boilerplate.

## 14.3 Hybrid Retrieval

Dense and sparse run **in parallel with identical payload filters**; candidate lists are merged by **Reciprocal Rank Fusion** — $\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$ with $k \approx 60$ — chosen because it is scale-free (no score normalization fragility) and consistently robust across query types. A weighted-score fusion mode ($\alpha \cdot \text{dense} + (1-\alpha) \cdot \text{sparse}$, α default 0.6) exists behind configuration for evaluation ablations. Hybrid is the default retrieval mode for all user searches.

## 14.4 Reranking

The fused top-N (default 100) goes to a cross-encoder reranker that jointly encodes (query text, candidate text) — full attention across both — yielding far finer relevance than bi-encoder similarity. Candidate text for reranking = the case's best-matching passages + compact entity summary. Output: rerank scores that (a) reorder the list, (b) feed confidence calibration, (c) are compared against pre-rerank order in the evaluation ablations to prove their contribution. Rerank depth trades latency for quality per retrieval profile (fast: skip rerank / top-50; balanced: top-100; thorough: top-200 + facet-level rerank).

## 14.5 Top-K Retrieval

Final K is user-configurable (10 default, 25 max in UI; researchers may export larger cohorts). Grouping guarantees one entry per case. Each entry carries: fused score, rerank score, per-facet similarity decomposition, matched entities, confidence label (high/moderate/weak with basis), and quality flags inherited from ingestion.

## 14.6 Metadata Filtering

Filters apply at three layers: (1) **user-selected** (demographics, codes, years, departments, outcome classes, quality exclusions) via UI or parsed from natural language; (2) **system-injected** (permitted datasets, active embedding version) — mandatory, server-side; (3) **soft preferences** (e.g., prefer same sex) implemented as score adjustments rather than hard filters when the user marks them "preferred" instead of "required". All filters execute inside the ANN query (pre-filtering), and the applied filter set is echoed in the response and the audit log for reproducibility.

---

# 15. Medical AI & Clinical NLP

## 15.1 Clinical Entities

The extraction ontology covers: **problems/diagnoses, signs & symptoms, medications (+dose, route, frequency, status), laboratory tests (+value, unit, flag, time), procedures, anatomical sites, temporal expressions, outcome expressions** (improved, discharged, expired, readmitted), and **risk factors** (smoking, alcohol, family history). Every entity: text span, section, type, assertion status, confidence, normalized code(s).

## 15.2 Symptoms

Symptom mentions are extracted with severity ("severe", "mild"), laterality, duration, and assertion. Negation handling is critical: "denies chest pain, no fever" produces _negated_ entities that (a) are excluded from positive-symptom facets, (b) can still participate in similarity as explicit negatives ("both cases notably lack chest pain despite cardiac presentation"). Symptom vocabulary normalizes to SNOMED CT findings.

## 15.3 Diseases

Disease/diagnosis mentions are resolved to **ICD-10** (billing/filter standard) and **SNOMED CT** (clinical granularity + hierarchy). SNOMED's hierarchy powers graded similarity: "NSTEMI" and "STEMI" are siblings under acute MI — closer than either is to "stable angina" — enabling ontology-aware diagnosis-overlap scoring in facet decomposition, not just exact-code matching.

## 15.4 ICD-10

Used for: filterable payload codes (with prefix matching, e.g., `E11*` = all T2DM), analytics chapters, cohort definitions, and interoperability with hospital systems. Mapping strategy: direct lexicon match → SNOMED-to-ICD map → contextual disambiguation model for ambiguous mentions; unmapped mentions flagged and reported in dataset quality stats.

## 15.5 SNOMED CT

The clinical backbone vocabulary: fine-grained concepts, is-a hierarchy, and relationship types. Used for normalization targets, hierarchy-based similarity, and abbreviation disambiguation support. Deployment note: SNOMED requires licensing per affiliate territory — the terminology layer is pluggable so deployments without SNOMED degrade gracefully to ICD-10 + UMLS-derived synonyms.

## 15.6 LOINC

Every lab analyte maps to LOINC; units normalize to UCUM; values are compared against reference ranges to compute abnormality flags (H/L/critical). This enables: structured lab filters ("abnormal creatinine"), lab-trend timelines, and the lab-pattern similarity facet (vectorized abnormality profiles + embedded lab summaries).

## 15.7 Drug Names

Medication mentions (brand, generic, misspelled, abbreviated) normalize to **RxNorm** ingredient level (brand→ingredient collapse so "Lasix" ≡ "furosemide"). Attributes extracted: dose, unit, route, frequency, and status (active/discontinued/held). Drug-class rollups (ATC) power class-level matching ("both on loop diuretics") in explanations and filters.

## 15.8 Lab Values

Numeric parsing handles formats ("1.8 mg/dL", "1,8 mg/dl", "creat 1.8"), ranges, and qualitative results ("positive", "trace"). Each result: analyte (LOINC), value, unit (UCUM-normalized), flag, timestamp (normalized), specimen where present, and provenance. OCR-derived values from low-confidence table regions carry a quality flag and are excluded from strict numeric filters by default.

## 15.9 Medical Abbreviations

A curated clinical abbreviation lexicon (thousands of entries) plus a context disambiguation step: candidate expansions scored by section type and surrounding entities ("PE" in a respiratory A&P with d-dimer context → pulmonary embolism; in a physical exam header → physical examination). Expansion decisions are stored with confidence; ambiguous cases retain both the original token and top expansion so nothing is destroyed.

## 15.10 Normalization

The unifying step (§8.5): mention → candidate concepts (lexical + embedding-based candidate generation against the terminology cache) → disambiguation (context ranking) → final code(s) with confidence. All normalization is non-destructive: original text is always preserved alongside codes; normalization model/lexicon versions are stamped for reproducibility; per-dataset mapping-rate dashboards expose coverage gaps.

## 15.11 Temporal Modeling & Timeline

Temporal expressions ("3 days prior to admission", "since 2019", "post-op day 2") normalize to absolute or admission-relative anchors. Events (symptom onset, dx, med start/stop, labs, procedures, outcomes) are ordered into the case timeline with confidence; unresolvable orderings are flagged rather than guessed. Timelines power the Timeline view, trajectory-aware comparison, and "events within N days of index event" filters.

---

# 16. Frontend Specification

Design language: clinical, calm, information-dense but uncluttered; light/dark themes; WCAG 2.1 AA; desktop-first with tablet support; every AI output visually distinguishes _evidence_ (neutral) from _AI narrative_ (labeled) and shows confidence.

## 16.1 Landing Page

Public marketing/entry page: product narrative (the §2 vision in user language), CDSS positioning and explicit "not a diagnostic device" statement, feature highlights with pipeline visual, security/compliance summary, and sign-in entry. No product data exposed.

## 16.2 Login

Credential + MFA flow, SSO button when configured, password reset, lockout messaging (non-enumerating), session-expired return-to-intent. Brand-consistent, minimal, fast.

## 16.3 Dashboard

Post-login home: quick search bar (natural language), recent search sessions, saved searches with change badges ("2 new similar cases"), ingestion status summary (for data roles), personal usage stats, and platform notices. Role-adaptive: physicians see search-centric layout; data managers see pipeline-centric layout.

## 16.4 Patient Search

The core screen. Two query modes: (a) **case-based** — pick/upload a patient record as the query; (b) **text-based** — natural-language clinical description. Filter rail: demographics, ICD prefix picker with autocomplete, medications, labs abnormal, year range, department, outcome class, quality exclusions, retrieval profile (fast/balanced/thorough). Results: ranked case cards — headline summary, overall score bar, confidence label, facet-score mini-bars (symptoms/dx/meds/labs/history), matched-entity chips, and "Explain" (lazy LLM explanation), "Compare", "Open", "More like this" actions. Parsed natural-language constraints are displayed as removable chips so the user sees exactly how their query was interpreted.

## 16.5 Patient Viewer

Full case view: structured summary header (demographics, key dx, outcome class), tabbed sections (narrative sections with entity highlighting, entities table with codes and confidence, labs with trend sparklines, documents with original-file viewer + OCR overlay), and provenance interactions — click any structured fact to see its highlighted source span, including OCR bounding boxes on scanned pages. If opened from a search, a persistent "similarity panel" shows why this case matched, with citations that jump to highlighted spans.

## 16.6 Medical Timeline

Interactive horizontal timeline per case: typed event lanes (symptoms, diagnoses, medications with duration bars, labs as value dots with abnormality coloring, procedures, encounters, outcomes), zoom/pan, event-type toggles, lab-series overlay charts, uncertainty styling for low-confidence anchors, and an optional LLM chronological narrative (labeled, cited).

## 16.7 Case Comparison

Side-by-side (2–3 cases): aligned timelines (align on admission/index event), entity diff view (shared vs. unique conditions/meds/procedures as a Venn-style list), lab comparison charts (same analyte overlaid), outcome panel, and the LLM comparative narrative with citations. Export to report from here.

## 16.8 Analytics

The §7.16 dashboard: archive composition, code distributions, demographic and outcome distributions, usage analytics, and cohort analytics for a selected search session — all ECharts, all drill-down-to-records (permission-aware), all with visible sample sizes and descriptive-statistics caveats.

## 16.9 Settings

Profile, password/MFA management, active sessions with revocation, notification preferences, UI preferences (theme, default retrieval profile, default K), and personal API token management for researcher exports (if enabled by admin).

## 16.10 Admin

The §7.18 panel: user/role management, dataset permissions, pipeline configuration, prompt template activation, job queue console (retry/requeue/dead-letter), feature flags, retention/export policy, and system health summary. Every admin mutation shows a confirmation with impact summary and is audit-logged.

## 16.11 Evaluation Dashboard

The §7.17 surface: retrieval metric trends with ablation toggles (dense/sparse/hybrid/reranked), extraction F1 per entity type, groundedness and citation-validity trends, latency/cost panels, gold-set management (upload judgments, annotation queue), and configuration-version comparison views ("v3 vs v4 collection: ΔnDCG@10 +0.04").

---

# 17. User Flow

**1. Login.** Physician authenticates (password + TOTP). Session established; role-scoped dashboard loads. _Audit: login event._

**2. Upload patient.** From the dashboard, "New case search" → upload de-identified files (drag-drop: scanned referral PDF + note DOCX + labs CSV) or paste a clinical description. Upload validates instantly; a processing tracker appears. _Audit: upload events._

**3. AI processing.** Live pipeline progress (OCR → parsing → extraction → embedding), typically seconds to ~2 minutes for scanned multi-page files. On completion, the extracted structured summary is shown for **user verification** — the physician can correct obvious extraction errors (e.g., mis-OCR'd lab value) before searching; corrections are versioned. _Audit: processing + edits._

**4. Retrieve similar patients.** Physician optionally adjusts filters, hits Search. Hybrid retrieval returns the ranked cohort in ≤5 s with scores, facet breakdowns, confidence labels, and interpreted-query chips. _Audit: query, filters, result IDs, scores._

**5. Review explanations.** For interesting results, "Explain" renders the cited LLM rationale; citation clicks open highlighted source passages. Physician marks results relevant/irrelevant (feeding evaluation). Weak matches are visibly labeled with their basis. _Audit: views, feedback, generations._

**6. Compare outcomes.** Physician selects 2–3 cases → Comparison view: aligned timelines, entity diffs, lab overlays, and the cohort Outcome Analysis panel ("of 25 similar cases: treatment paths × documented outcomes"), all descriptive, all cited, all caveated. _Audit: comparison views._

**7. Export report.** "Export session report" → cited PDF: query summary, retrieved cohort with scores, explanations, comparison highlights, outcome table, generation metadata (model/template versions), timestamp, and the mandatory CDSS disclaimer. Stored in the session history. _Audit: export event._

The loop closes: feedback from step 5 becomes weak labels in the Evaluation Dashboard, driving retrieval-quality improvement over time.

---

# 18. Technology Stack

| Layer          | Technology                                         | Rationale                                                                                                                                                |
| -------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend API    | **FastAPI** (Python)                               | Async-first for I/O-heavy retrieval orchestration; Pydantic validation everywhere; first-class OpenAPI; the Python ecosystem is where the AI stack lives |
| Frontend       | **React** (+ TypeScript)                           | Component ecosystem, mature state/data-fetching patterns, ECharts bindings                                                                               |
| Relational DB  | **PostgreSQL**                                     | Integrity, migrations, partitioning, JSONB flexibility for entity payloads, boring reliability                                                           |
| Vector DB      | **Qdrant**                                         | Filtered HNSW ANN, named dense+sparse vectors on one point, payload indexes, snapshots, hybrid Query API, self-hostable in Docker                        |
| Cache/Queue    | **Redis**                                          | Caches, job queues, rate limiting, locks — one dependable tool for all ephemeral state                                                                   |
| LLM            | **Gemini**                                         | Long context (large retrieval payloads), strong structured-output adherence, competitive latency/cost                                                    |
| Embeddings     | **BGE-M3**                                         | Dense+sparse+multi-vector in one model, 8k context, multilingual, self-hostable                                                                          |
| Reranker       | BGE-reranker class                                 | Cross-encoder precision for stage 9                                                                                                                      |
| OCR            | **PaddleOCR** (PP-OCR/PP-Structure)                | Strong accuracy, table/layout analysis, self-hosted (PHI never leaves infrastructure), GPU-optional                                                      |
| Charts         | **Apache ECharts**                                 | Clinical-grade dense visualizations: timelines, heatmaps, trend overlays                                                                                 |
| Jobs           | Python workers over Redis queues                   | Simple, observable, horizontally scalable pipeline stages                                                                                                |
| Object storage | S3-compatible (MinIO/S3)                           | Content-addressed originals, exports                                                                                                                     |
| Deployment     | **Docker** / Docker Compose (v1), Kubernetes-ready | Reproducible dev/prod parity; hospital-friendly self-hosting                                                                                             |
| Observability  | Prometheus + Grafana + structured logs + tracing   | §7.20                                                                                                                                                    |

Principle: self-hostable by default — the only external dependency touching case-derived text is the LLM API, and it receives only de-identified, minimum-necessary context (with a documented path to a self-hosted LLM for strict deployments).

---

# 19. Software Architecture

## 19.1 Overview

A modular service architecture — a well-factored FastAPI application plus dedicated worker processes — deliberately avoiding premature microservices while keeping seams clean enough to split later.

```
┌────────────┐   HTTPS    ┌─────────────────────────────┐
│  React SPA │ ─────────▶ │        API Gateway           │
└────────────┘            │  (FastAPI: auth, RBAC,       │
                          │   rate limits, audit hooks)  │
                          └──────┬──────────────┬───────┘
                                 │              │ enqueue
                    sync reads/writes           ▼
                                 │      ┌───────────────┐     ┌──────────────┐
                                 │      │ Redis (queues,│────▶│ Worker Fleet │
                                 │      │ cache, locks) │     │ OCR │ Parse  │
                                 ▼      └───────────────┘     │ NLP │ Embed  │
        ┌────────────────────────────────────┐                │ Index│Explain│
        │        AI Services Layer           │◀───────────────┴──────────────┘
        │ retrieval orchestrator · reranker  │
        │ context builder · LLM gateway      │
        └───────┬───────────────┬────────────┘
                ▼               ▼                      ▼
        ┌────────────┐   ┌────────────┐        ┌─────────────┐
        │ PostgreSQL │   │   Qdrant   │        │ Object Store│
        │  (truth)   │   │ (similarity)│       │ (originals) │
        └────────────┘   └────────────┘        └─────────────┘
                    ▲ metrics/logs/traces from every box
              ┌─────┴──────────────────────────────┐
              │  Monitoring: Prometheus · Grafana  │
              └────────────────────────────────────┘
```

## 19.2 Frontend

React SPA served statically (nginx container). Talks only to the API over HTTPS/JSON; receives live ingestion progress via WebSocket/SSE. No business logic client-side beyond presentation; all authorization re-checked server-side.

## 19.3 API Layer

FastAPI application organized by module (§7): routers → services → repositories. Cross-cutting middleware: authentication, RBAC enforcement, rate limiting (Redis), request validation (Pydantic), correlation-ID injection, audit hooks, and uniform error envelopes. Synchronous paths (search, views) are async-I/O; anything heavy (OCR, embedding, generation batches) is enqueued.

## 19.4 AI Services

Internal service layer (same codebase, isolated packages) hosting: retrieval orchestrator (dense/sparse/fusion/filters), reranker service (model-serving, GPU-optional), context builder, LLM gateway (templates, validation, budgets, caching, provider abstraction), and NLP components. Each exposes clean internal interfaces so any one can be split into its own deployment when scale demands.

## 19.5 Databases

As specified in §10: PostgreSQL (truth), Qdrant (similarity, rebuildable), Redis (ephemeral), object storage (binaries). Backup posture: PostgreSQL PITR + nightly dumps; Qdrant snapshots + rebuildability; object store versioning.

## 19.6 Background Workers

One worker type per pipeline stage (intake, OCR, parse, NLP, embed, index, explain-precompute, reconcile, report-export), consuming Redis queues, horizontally scalable and independently resourced (OCR/embed workers get GPU; parse workers are CPU-cheap). Properties: idempotent handlers, bounded retries with exponential backoff, dead-letter queues with admin console, per-stage metrics, graceful shutdown (job re-queue on SIGTERM).

## 19.7 Storage

Object storage for originals/exports (content-addressed, encrypted); local scratch volumes for OCR intermediates (auto-cleaned); model files (embeddings, reranker, OCR) baked into worker images or pulled from a model registry volume at startup with checksum verification.

## 19.8 Monitoring

As per §7.20 — metrics, logs, traces from every container; correlation IDs from browser request to LLM call; Grafana dashboards and alerting. The evaluation pipeline (§22) runs as scheduled jobs inside this same worker infrastructure.

---

# 20. Folder Structure

```
medsearch-ai/
├── README.md
├── PROJECT_SPECIFICATION.md
├── docker-compose.yml                 # full local stack
├── docker-compose.prod.yml
├── .env.example
├── Makefile                           # dev entrypoints: up, test, lint, seed, eval
│
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── alembic/                       # DB migrations
│   ├── app/
│   │   ├── main.py                    # FastAPI app factory
│   │   ├── core/                      # config, security, logging, correlation IDs
│   │   ├── api/
│   │   │   └── v1/                    # routers per module: auth, users, datasets,
│   │   │                              # documents, records, search, timeline,
│   │   │                              # compare, analytics, evaluation, admin, audit
│   │   ├── modules/                   # domain services + repositories per module
│   │   │   ├── auth/
│   │   │   ├── users/
│   │   │   ├── datasets/
│   │   │   ├── records/
│   │   │   ├── search/
│   │   │   ├── audit/
│   │   │   └── admin/
│   │   ├── db/                        # SQLAlchemy models, session, base repos
│   │   ├── schemas/                   # Pydantic request/response models
│   │   └── ws/                        # WebSocket/SSE progress channels
│   └── tests/                         # unit + integration (mirrors app/)
│
├── ai/
│   ├── Dockerfile.worker              # CPU worker image
│   ├── Dockerfile.worker-gpu          # GPU worker image (OCR, embed, rerank)
│   ├── pipeline/
│   │   ├── intake/                    # validation, dedup, quarantine
│   │   ├── ocr/                       # PaddleOCR wrappers, preprocessing, tables
│   │   ├── parsing/                   # pdf/docx/txt/fhir/hl7/csv parsers, sectionizer
│   │   ├── deid/                      # de-identification stage
│   │   ├── nlp/                       # NER, assertion, relations, temporal
│   │   ├── normalization/             # ICD-10/SNOMED/LOINC/RxNorm mapping
│   │   ├── embedding/                 # chunking, BGE-M3 inference, caching
│   │   └── indexing/                  # Qdrant upserts, registry, reconciliation
│   ├── retrieval/
│   │   ├── dense.py  sparse.py  fusion.py  rerank.py  filters.py  profiles.py
│   ├── llm/
│   │   ├── gateway/                   # provider abstraction, budgets, caching
│   │   ├── prompts/                   # versioned templates (one file per version)
│   │   ├── context/                   # context builder, token budgeting
│   │   └── validation/                # schema, citations, groundedness, phrases
│   ├── terminology/                   # vocab loaders, abbreviation lexicon
│   ├── workers/                       # queue consumers per stage
│   └── tests/                         # incl. prompt regression tests, gold fixtures
│
├── evaluation/
│   ├── goldsets/                      # query sets + relevance judgments (versioned)
│   ├── runners/                       # retrieval/extraction/generation eval jobs
│   ├── metrics/                       # recall@k, ndcg, mrr, f1, groundedness
│   └── reports/                       # generated metric snapshots
│
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   ├── src/
│   │   ├── app/                       # routing, providers, theming
│   │   ├── api/                       # typed API client, hooks
│   │   ├── components/                # shared UI (cards, chips, citation viewer)
│   │   ├── features/
│   │   │   ├── auth/  dashboard/  search/  patient-viewer/  timeline/
│   │   │   ├── compare/  analytics/  evaluation/  settings/  admin/
│   │   ├── charts/                    # ECharts wrappers (timeline, trends, dists)
│   │   └── utils/
│   └── tests/
│
├── infra/
│   ├── nginx/                         # reverse proxy, TLS, SPA serving
│   ├── postgres/                      # init, tuning
│   ├── qdrant/                        # collection bootstrap config
│   ├── monitoring/                    # prometheus, grafana dashboards, alerts
│   └── scripts/                       # backup, restore, reindex, seed
│
├── docs/
│   ├── architecture/                  # ADRs (Architecture Decision Records)
│   ├── pipeline/                      # stage-by-stage docs
│   ├── security/                      # threat model, data-handling policy
│   └── runbooks/                      # ops procedures
│
└── .github/
    └── workflows/                     # CI: lint, tests, eval regression, build
```

Conventions: every module owns its router/service/repository/tests vertically; prompts and gold sets are versioned artifacts under review like code; ADRs record every consequential design decision.

---

# 21. Development Roadmap

Each phase ends with a demoable milestone and defined exit criteria. Phases build strictly on one another.

## Phase 1 — Foundation

**Scope:** repository scaffold and conventions; Docker Compose stack (PostgreSQL, Redis, Qdrant, MinIO, API, frontend shell); FastAPI skeleton with config, logging, correlation IDs, error envelopes; Authentication (JWT + refresh + lockout) and User Management (RBAC roles, invitations); base schema migrations; Audit Log core (append-only writes on auth/user events); CI (lint, tests, build); monitoring baseline (metrics endpoint, Grafana up).
**Exit criteria:** a user can be invited, log in with MFA, and every auth event appears in the audit log; the full stack boots with one command.

## Phase 2 — Data Pipeline

**Scope:** Medical Document Upload (chunked, validated, content-addressed); Dataset Management (datasets, batches, states); worker framework over Redis queues (idempotency, retries, DLQ); OCR stage (PaddleOCR + preprocessing + tables + confidence gating); Document Parsing for all §9 formats; cleaning + sectionizer; de-identification stage; ingestion progress UI; pipeline observability.
**Exit criteria:** a mixed batch (scanned PDF, DOCX, CSV, FHIR) ingests end-to-end into normalized document models with per-document status, provenance, and quality flags.

## Phase 3 — Embeddings _(includes Medical NLP)_

**Scope:** clinical NER, assertion classification, relation extraction, temporal anchoring; normalization to ICD-10/SNOMED/LOINC/RxNorm with terminology cache; structured Patient Records with provenance; facet text derivation; chunking policy; BGE-M3 serving (dense+sparse) with batching and embedding cache; extraction quality harness against a small annotated set.
**Exit criteria:** ingested documents produce coded, versioned patient records and versioned vectors; extraction F1 baseline measured and displayed.

## Phase 4 — Vector Search

**Scope:** Qdrant collections (named dense+sparse vectors, payload schema, payload indexes); indexing service with deterministic IDs and reconciliation; dense retrieval; sparse retrieval; RRF hybrid fusion; metadata filtering (user + injected); group-by-case; Search API + Patient Search UI (cards, scores, facet breakdowns, filters); search sessions and audit of every query; first gold query set and Recall@K/nDCG baseline in a minimal evaluation report.
**Exit criteria:** hybrid filtered search over a seeded corpus (e.g., MIMIC-derived demo set) returns ranked cases in <2 s with measured retrieval metrics.

## Phase 5 — LLM

**Scope:** LLM gateway (Gemini, budgets, caching, provider abstraction); context builder with token budgeting and source tagging; prompt templates v1 (similarity explanation, comparison, cohort outcomes, query understanding); structured-output schemas and validation (citations, groundedness, banned phrases, numeric verification); reranker deployment and integration; confidence calibration; graceful degradation path; explanation UI with clickable citations.
**Exit criteria:** every viewed result can render a validated, cited explanation; groundedness ≥95% on the test suite; reranking demonstrably improves nDCG on the gold set.

## Phase 6 — Frontend

**Scope:** complete the product surface — Patient Viewer with provenance highlighting and OCR overlays; Medical Timeline (interactive, lab trends); Case Comparison (aligned timelines, entity diff, outcome panel); Analytics dashboard (ECharts); natural-language query chips; Settings; Admin panel (users, datasets, pipeline config, job console); report export (cited PDF); accessibility pass; dark mode.
**Exit criteria:** the full §17 user flow runs end-to-end through the UI without touching an API client.

## Phase 7 — Evaluation

**Scope:** Evaluation Dashboard (retrieval trends, ablations dense/sparse/hybrid/reranked, extraction F1, groundedness, latency/cost); gold-set management and annotation queue; physician feedback capture wired into weak labels; scheduled evaluation runs with regression alerts in CI; configuration-version comparison (collection v(n) vs v(n+1)); load testing against latency targets; retrieval profile tuning.
**Exit criteria:** quality metrics from §3.3 are continuously measured, visible, and alarmed; an embedding/config change can be A/B evaluated and promoted or rolled back with evidence.

## Phase 8 — Deployment

**Scope:** production Compose/K8s manifests; TLS, secrets management, network policy; backups and restore drills (PostgreSQL PITR, Qdrant snapshots, object versioning); blue/green reindex runbook; security hardening review (OWASP pass, dependency audit, container scanning); rate limiting and quota polish; documentation (runbooks, admin guide, data-handling policy); demo dataset and guided demo script; final performance validation against §3.3 targets.
**Exit criteria:** a clean machine goes from zero to a secured, monitored, seeded production deployment using documented procedures; all §3.3 targets verified.

---

# 22. Evaluation & Quality Strategy

Quality is proven, not assumed:

- **Gold sets.** Curated query→relevant-case judgment sets (built via annotation queue + physician feedback), versioned in the repo; extraction gold set of annotated documents per entity type.
- **Retrieval evaluation.** Recall@K, Precision@K, MRR, nDCG@10 computed per configuration; standing ablation: dense-only vs sparse-only vs hybrid vs hybrid+rerank — the ablation _is_ the justification for the architecture.
- **Extraction evaluation.** Per-entity-type P/R/F1; normalization accuracy; negation accuracy; OCR-quality-stratified reporting.
- **Generation evaluation.** Automated groundedness scoring, citation validity rate, banned-phrase incidence, refusal correctness on an adversarial prompt suite ("what's the diagnosis?", "what should I prescribe?"), plus sampled human review with rubric.
- **Regression discipline.** Prompt and pipeline changes run the evaluation suite in CI; metric regressions block promotion.
- **Feedback loop.** Accept/dismiss/flag signals accumulate as weak labels, periodically promoted to gold via annotation review.

---

# 23. Security, Privacy & Compliance

- **Data protection.** De-identification as a mandatory pipeline stage; date shifting preserving intervals; age bands instead of birth dates in indexes; encryption in transit (TLS everywhere) and at rest (DB, object store, backups); content minimization to the LLM (de-identified, need-to-know context only).
- **Access control.** RBAC with dataset-level grants; deny-by-default; server-side enforcement on every request and every vector query (injected filters); admin actions double-confirmed.
- **Application security.** OWASP Top 10 addressed by design: parameterized queries only, strict input validation (Pydantic), output encoding, file-type sniffing and sanitization, SSRF-safe fetchers, dependency and container scanning in CI, secrets in a manager (never in code or images), security headers, and rate limiting.
- **Auditability.** §7.19 — complete, tamper-evident, retained, and itself access-controlled.
- **Regulatory posture.** Designed to operate as non-diagnostic CDSS with human-in-the-loop; HIPAA-aligned technical safeguards (access control, audit, integrity, transmission security) and GDPR-aligned rights support (deletion flow §12.9, purpose limitation, data minimization). Formal certification activities are deployment-specific and out of v1 scope but nothing in the design obstructs them.
- **Safety guardrails.** Non-diagnostic constraint enforced at prompt, validator, and UI layers; permanent CDSS disclaimer on results and exports; adversarial-prompt test suite in CI.

---

# 24. Non-Functional Requirements

| Category             | Requirement                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| Performance          | Search p95 < 5 s end-to-end (§3.3); cached repeat search p95 < 1 s; explanation generation p95 < 8 s |
| Scale                | 1M+ indexed cases, 10M+ vector points per deployment; 100 concurrent users                           |
| Availability         | 99.5% search API; ingestion may degrade to queued mode without data loss                             |
| Durability           | Zero acknowledged-upload loss; PostgreSQL PITR; rebuildable Qdrant                                   |
| Reproducibility      | Every search and generation reproducible from logged versions + hashes                               |
| Portability          | Full stack runs on a single Docker host (dev) and scales out (prod)                                  |
| Accessibility        | WCAG 2.1 AA on all clinical screens                                                                  |
| Internationalization | UTF-8 throughout; multilingual embedding support; UI i18n-ready                                      |
| Maintainability      | Module boundaries per §7; ADRs; ≥80% test coverage on domain logic; prompt/eval regression gates     |

---

# 25. Risk Register

| Risk                                     | Likelihood | Impact   | Mitigation                                                                              |
| ---------------------------------------- | ---------- | -------- | --------------------------------------------------------------------------------------- |
| LLM hallucination reaches clinicians     | Medium     | Critical | Layered grounding + validation + degradation (§13.5); groundedness alarms               |
| Users treat output as diagnosis          | Medium     | Critical | Non-diagnostic framing everywhere, disclaimers, refusal behaviors, training material    |
| Poor OCR corrupts downstream data        | High       | High     | Confidence gating, quality flags propagated, human verification step (§17.3)            |
| Entity extraction errors skew similarity | Medium     | High     | Confidence-weighted facets, extraction evaluation, user correction loop                 |
| PHI leakage into index or LLM            | Low        | Critical | Mandatory de-id stage, payload design without identifiers, context minimization, audits |
| Retrieval quality unproven               | Medium     | High     | Gold sets + ablation dashboard from Phase 4 onward                                      |
| Vector/metadata drift                    | Medium     | Medium   | Deterministic IDs, reconciliation job, versioned collections                            |
| Embedding model change degrades quality  | Medium     | Medium   | Blue/green collections, A/B evaluation before promotion, instant rollback               |
| Cost blowout on LLM calls                | Medium     | Medium   | Lazy per-view generation, caching, token budgets, cost dashboards                       |
| Terminology licensing gaps (SNOMED)      | Medium     | Medium   | Pluggable terminology layer with graceful ICD-10-only degradation                       |

---

# 26. Future Features

Post-v1 roadmap, in rough priority order. The v1 architecture deliberately leaves seams for each.

1. **Medical Image Search.** Extend retrieval to imaging: image embedding models index radiology images; "find cases with similar imaging" joins text-based similarity. New Qdrant collections; same retrieval orchestration.
2. **Chest X-Ray Embeddings.** Dedicated CXR encoder (CheXzero/BiomedCLIP class) producing image vectors linked to their report vectors — enabling text↔image cross-modal retrieval ("cases whose films look like this one").
3. **CT Scan Embeddings.** Volumetric (3D) encoders for CT series; slice-level and study-level vectors; heavier GPU and storage profile, isolated in its own worker class.
4. **ECG Analysis.** Waveform encoders embed 12-lead ECGs; rhythm/morphology similarity search joins the facet family ("similar presentation _and_ similar ECG").
5. **Knowledge Graph.** Promote normalized entities and relations into an explicit clinical knowledge graph (patient–condition–treatment–outcome), enabling graph-constrained retrieval, path-based explanations ("similar via shared comorbidity cluster"), and cohort discovery beyond vector proximity.
6. **Multi-Agent AI.** Decompose complex queries across cooperating agents: a retrieval agent, a cohort-statistics agent, a literature agent, and a critique agent that audits groundedness before response assembly — all within the same evidence-only boundaries.
7. **Federated Search.** Query across multiple hospital deployments without moving patient data: queries travel, records don't; per-site retrieval with privacy-preserving score aggregation and strict k-anonymity thresholds on returned cohorts.
8. **Clinical Trial Matching.** Index trial eligibility criteria as structured + embedded documents; match the query patient's entities against criteria to surface potentially relevant trials (with the same evidence-only, clinician-confirms framing).
9. **Drug Recommendation (evidence surfacing).** Not prescribing — surfacing: for a retrieved cohort, structured views of medication-outcome associations with strength-of-evidence labels and literature cross-references, framed strictly as historical observation.
10. **Multimodal Retrieval.** Unified case representation fusing text, labs-as-timeseries, images, and waveforms into joint or late-fused similarity — the long-term destination: "find patients like this one" across every modality at once.

---

# 27. Glossary

| Term             | Definition                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| ANN              | Approximate Nearest Neighbor search                                                                                      |
| Assertion status | Whether a clinical mention is affirmed, negated, uncertain, historical, or about family                                  |
| BGE-M3           | Embedding model producing dense, sparse, and multi-vector representations                                                |
| CDSS             | Clinical Decision Support System — assists, never replaces, clinician judgment                                           |
| Cross-encoder    | Reranking model jointly encoding query and candidate for high-precision scoring                                          |
| Dense retrieval  | Similarity search over semantic embedding vectors                                                                        |
| Facet            | A clinical dimension of a case (symptoms, diagnoses, medications, labs, history, outcome) embedded and scored separately |
| FHIR             | Fast Healthcare Interoperability Resources — healthcare data exchange standard                                           |
| Groundedness     | Degree to which generated text is attributable to supplied source context                                                |
| HL7 v2           | Legacy healthcare messaging standard                                                                                     |
| HNSW             | Hierarchical Navigable Small World — graph index for fast ANN                                                            |
| Hybrid search    | Combined dense (semantic) + sparse (lexical) retrieval with fusion                                                       |
| ICD-10           | International Classification of Diseases, 10th revision                                                                  |
| LOINC            | Logical Observation Identifiers Names and Codes — lab test vocabulary                                                    |
| NER              | Named Entity Recognition                                                                                                 |
| nDCG             | Normalized Discounted Cumulative Gain — ranking quality metric                                                           |
| PHI              | Protected Health Information                                                                                             |
| RAG              | Retrieval-Augmented Generation                                                                                           |
| RRF              | Reciprocal Rank Fusion — rank-based result list merging                                                                  |
| RxNorm           | Normalized vocabulary for medications                                                                                    |
| SNOMED CT        | Comprehensive clinical terminology with concept hierarchy                                                                |
| Sparse retrieval | Lexical-weight-based retrieval (exact/rare term precision)                                                               |
| UCUM             | Unified Code for Units of Measure                                                                                        |

---

_End of specification. This document is the authoritative reference for MedSearch AI. Any implementation decision that conflicts with it requires an ADR amending the relevant section._
