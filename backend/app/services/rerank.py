"""Rerank service — NVIDIA-hosted cross-encoder for "thorough mode".

Calls the NIM reranking API (llama-nemotron-rerank-1b-v2) on the top
retrieval candidates and blends the reranker's relevance logits with the
retrieval scores. Measured on the gold set (evaluation/RESULTS.md):
nDCG@10 0.942 / Recall@1 0.929 at depth=50, beta=0.9 — vs 0.658 nDCG for
retrieval alone. Mean API latency ~0.33s.

Fails soft: any API error returns the original retrieval order.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import requests

from app.core.config import settings
from app.schemas.api import CaseResult

logger = logging.getLogger(__name__)


class RerankService:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._docs: dict[str, str] | None = None  # case_id -> first N chars
        self._lock = threading.Lock()

    def available(self) -> bool:
        return bool(settings.nvidia_api_key)

    def warm(self) -> None:
        """Build the doc cache off the critical path (called at startup in a
        background thread — the bind-mounted file read takes ~50s)."""
        self._doc_cache()

    def docs_ready(self) -> dict[str, str] | None:
        """Non-blocking view of the doc cache (None while still warming).
        Used by the negation layer, which must never wait on the lock."""
        return self._docs

    def _doc_cache(self) -> dict[str, str]:
        """Trimmed case texts, loaded once (~40MB). Avoids per-query file seeks
        which are slow on bind-mounted volumes."""
        with self._lock:
            if self._docs is None:
                t0 = time.time()
                docs: dict[str, str] = {}
                with settings.cases_file.open(encoding="utf-8") as fh:
                    for line in fh:
                        rec = json.loads(line)
                        docs[rec["case_id"]] = rec["document"][: settings.rerank_doc_chars]
                self._docs = docs
                logger.info("rerank doc cache: %d cases in %.1fs", len(docs), time.time() - t0)
        return self._docs

    def rerank(self, query: str, results: list[CaseResult], k: int) -> list[CaseResult]:
        """Reorder `results` (retrieval order, up to rerank_depth) and return top k."""
        if not results:
            return results
        t0 = time.time()
        docs = self._doc_cache()
        t_docs = time.time() - t0
        texts = [docs.get(r.case_id) or r.snippet for r in results]
        try:
            t1 = time.time()
            logits = self._score(query, texts)
            logger.info("rerank timing: docs=%.2fs api=%.2fs n=%d", t_docs, time.time() - t1, len(texts))
        except Exception:
            logger.exception("rerank API failed — falling back to retrieval order")
            return results[:k]

        def norm(vals: list[float]) -> list[float]:
            lo, hi = min(vals), max(vals)
            rng = (hi - lo) or 1.0
            return [(v - lo) / rng for v in vals]

        rr = norm(logits)
        rt = norm([r.score for r in results])
        beta = settings.rerank_beta
        blended = [beta * a + (1 - beta) * b for a, b in zip(rr, rt)]
        order = sorted(range(len(results)), key=lambda i: -blended[i])

        out = []
        for i in order[:k]:
            r = results[i]
            r.score = round(blended[i], 4)
            out.append(r)
        return out

    def _score(self, query: str, texts: list[str]) -> list[float]:
        resp = self._session.post(
            settings.rerank_url,
            json={
                "model": settings.rerank_model,
                "query": {"text": query},
                "passages": [{"text": t} for t in texts],
                "truncate": "END",
            },
            headers={
                "Authorization": f"Bearer {settings.nvidia_api_key}",
                "Accept": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        scores = [0.0] * len(texts)
        for item in resp.json()["rankings"]:
            scores[item["index"]] = float(item["logit"])
        return scores


rerank_service = RerankService()
