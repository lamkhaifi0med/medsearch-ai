"""Retrieval service — query embedding + hybrid search (spec §7.12–7.13, §14).

The BGE-M3 model loads ONCE at startup and stays warm; each query then costs
~1 s of CPU embedding + <100 ms of Qdrant search.
"""

from __future__ import annotations

import logging
import time

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.schemas.api import CaseResult, SearchFilters

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self) -> None:
        self._client = QdrantClient(url=settings.qdrant_url, timeout=60)
        self._model = None

    def load_model(self) -> None:
        """Called once at app startup (lifespan)."""
        logger.info("Loading embedding model %s ...", settings.embed_model_name)
        t0 = time.time()
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(settings.embed_model_name, use_fp16=False)
        logger.info("Embedding model ready in %.1fs", time.time() - t0)

    # ------------------------------------------------------------ embedding

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        assert self._model is not None, "model not loaded"
        enc = self._model.encode([text], return_dense=True, return_sparse=True, max_length=512)
        dense = enc["dense_vecs"][0].tolist()
        sp = enc["lexical_weights"][0]
        sparse = models.SparseVector(
            indices=[int(i) for i in sp.keys()],
            values=[float(v) for v in sp.values()],
        )
        return dense, sparse

    # ------------------------------------------------------------ filters

    def _build_filter(self, f: SearchFilters) -> models.Filter:
        # embedding_version is always injected server-side (spec §14.6)
        must: list[models.Condition] = [
            models.FieldCondition(
                key="embedding_version",
                match=models.MatchValue(value=settings.embedding_version),
            )
        ]
        if f.sex:
            must.append(models.FieldCondition(key="sex", match=models.MatchValue(value=f.sex)))
        if f.outcome_class:
            must.append(
                models.FieldCondition(key="outcome_class", match=models.MatchValue(value=f.outcome_class))
            )
        if f.age_min is not None or f.age_max is not None:
            must.append(
                models.FieldCondition(key="age", range=models.Range(gte=f.age_min, lte=f.age_max))
            )
        return models.Filter(must=must)

    # ------------------------------------------------------------ search

    @staticmethod
    def _weighted_fuse(dense_pts: list, sparse_pts: list, alpha: float, k: int) -> list:
        """alpha * minmax(dense) + (1-alpha) * minmax(sparse) — the fusion mode
        that won the evaluation ablation (evaluation/RESULTS.md)."""

        def norm(pts) -> dict[str, float]:
            if not pts:
                return {}
            scores = [p.score for p in pts]
            lo, hi = min(scores), max(scores)
            rng = (hi - lo) or 1.0
            return {p.payload["case_id"]: (p.score - lo) / rng for p in pts}

        dn, sn = norm(dense_pts), norm(sparse_pts)
        by_id = {p.payload["case_id"]: p for p in [*dense_pts, *sparse_pts]}
        fused: dict[str, float] = {}
        for cid, s in dn.items():
            fused[cid] = alpha * s
        for cid, s in sn.items():
            fused[cid] = fused.get(cid, 0.0) + (1 - alpha) * s

        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        out = []
        for cid, score in top:
            p = by_id[cid]
            p.score = score  # expose the fused 0–1 score
            out.append(p)
        return out

    def search(self, query: str, k: int, filters: SearchFilters, depth: int | None = None) -> list[CaseResult]:
        """Hybrid search. `depth` overrides k for the fused list length
        (used by rerank to get a deeper candidate pool)."""
        dense, sparse = self.embed_query(query)
        qfilter = self._build_filter(filters)

        dense_res = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=dense,
            using="dense",
            query_filter=qfilter,
            limit=settings.prefetch_limit,
            with_payload=True,
        )
        sparse_res = self._client.query_points(
            collection_name=settings.qdrant_collection,
            query=sparse,
            using="sparse",
            query_filter=qfilter,
            limit=settings.prefetch_limit,
            with_payload=True,
        )
        points = self._weighted_fuse(
            dense_res.points, sparse_res.points, settings.fusion_alpha, depth or k
        )

        return [
            CaseResult(
                case_id=p.payload["case_id"],
                score=round(p.score, 4),
                sex=p.payload.get("sex", "unknown"),
                age=p.payload.get("age"),
                age_band=p.payload.get("age_band", "unknown"),
                outcome_class=p.payload.get("outcome_class", "unknown"),
                snippet=p.payload.get("snippet", ""),
                quality_flags=p.payload.get("quality_flags", []),
            )
            for p in points
        ]

    # ------------------------------------------------------------ health

    def healthy(self) -> tuple[bool, int | None]:
        try:
            info = self._client.get_collection(settings.qdrant_collection)
            return True, info.points_count
        except Exception:
            return False, None


retrieval_service = RetrievalService()
