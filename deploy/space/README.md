---
title: MedSearch AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Clinical case retrieval — hybrid search + reranking + RAG
---

# MedSearch AI — Clinical Case Retrieval Engine

Describe a patient in plain language, retrieve the most similar historical
cases from **24,348 real de-identified clinical notes** — with hybrid
dense+sparse retrieval (BGE-M3 + Qdrant), optional cross-encoder reranking
(NVIDIA NIM), negation-aware scoring (NegEx), and citation-grounded LLM
explanations. **nDCG@10 0.94** on a reproducible gold set.

> It retrieves evidence — it never diagnoses.

- Source code & full documentation: https://github.com/lamkhaifi0med/medsearch-ai
- Note: first startup takes a few minutes (embedding model load). Fast search
  answers in ~1–3 s; "Thorough" mode adds ~0.5 s of GPU reranking.
