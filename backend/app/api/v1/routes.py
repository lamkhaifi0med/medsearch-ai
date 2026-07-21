"""API v1 routes: search, explain, cases, health (spec §19.3)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.api import (
    CaseDetail,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.cases import case_store
from app.services.llm import llm_gateway
from app.services.negation import negation_service
from app.services.rerank import rerank_service
from app.services.retrieval import retrieval_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.time()
    use_rerank = req.rerank and rerank_service.available()
    if use_rerank:
        candidates = retrieval_service.search(
            req.query, req.k, req.filters, depth=settings.rerank_depth
        )
        # keep the full reranked pool so negation can still reorder it
        results = rerank_service.rerank(req.query, candidates, settings.rerank_depth)
    else:
        depth = settings.negation_pool if settings.negation_enabled else None
        results = retrieval_service.search(req.query, req.k, req.filters, depth=depth)
    if settings.negation_enabled:
        results = negation_service.adjust(req.query, results, rerank_service.docs_ready())
    results = results[: req.k]
    took = int((time.time() - t0) * 1000)
    logger.info("search k=%d results=%d rerank=%s took=%dms", req.k, len(results), use_rerank, took)
    return SearchResponse(
        query=req.query,
        filters=req.filters,
        results=results,
        took_ms=took,
        embedding_version=settings.embedding_version,
        reranked=use_rerank,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    t0 = time.time()
    documents = case_store.get_documents(req.case_ids)
    if not documents:
        raise HTTPException(status_code=404, detail="none of the requested case_ids exist")
    missing = set(req.case_ids) - set(documents)
    if missing:
        raise HTTPException(status_code=404, detail=f"unknown case_ids: {sorted(missing)}")

    metadata = {}
    for cid in req.case_ids:
        detail = case_store.get(cid)
        if detail:
            metadata[cid] = {"sex": detail.sex, "age": detail.age, "outcome_class": detail.outcome_class}

    response = llm_gateway.explain(req.query, documents, metadata)
    response.took_ms = int((time.time() - t0) * 1000)
    logger.info(
        "explain cases=%d model=%s degraded=%s cached=%s took=%dms",
        len(req.case_ids), response.model_used, response.degraded, response.cached, response.took_ms,
    )
    return response


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: str) -> CaseDetail:
    detail = case_store.get(case_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"case {case_id} not found")
    return detail


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_ok, points = retrieval_service.healthy()
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant=qdrant_ok,
        redis=llm_gateway.cache_healthy(),
        llm_configured=bool(settings.nvidia_api_key),
        points_indexed=points,
    )
