"""API request/response models (spec §19.3 — Pydantic validation everywhere)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Sex = Literal["male", "female"]
Outcome = Literal["improved", "deteriorated", "deceased", "unknown"]
Confidence = Literal["high", "moderate", "weak"]


# ---------------------------------------------------------------- search

class SearchFilters(BaseModel):
    sex: Sex | None = None
    outcome_class: Outcome | None = None
    age_min: int | None = Field(None, ge=0, le=120)
    age_max: int | None = Field(None, ge=0, le=120)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    k: int = Field(10, ge=1, le=25)
    filters: SearchFilters = SearchFilters()
    rerank: bool = False  # "thorough mode": NVIDIA-hosted cross-encoder pass


class CaseResult(BaseModel):
    case_id: str
    score: float
    sex: str
    age: int | None
    age_band: str
    outcome_class: str
    snippet: str
    quality_flags: list[str] = []


class SearchResponse(BaseModel):
    query: str
    filters: SearchFilters
    results: list[CaseResult]
    took_ms: int
    embedding_version: str
    reranked: bool = False


# ---------------------------------------------------------------- explain

class ExplainRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    case_ids: list[str] = Field(..., min_length=1, max_length=5)


class SimilarityFactor(BaseModel):
    factor: str
    detail: str
    citations: list[str]


class Difference(BaseModel):
    detail: str
    citations: list[str]


class TreatmentObserved(BaseModel):
    treatment: str
    outcome_note: str
    citations: list[str]


class CaseExplanation(BaseModel):
    case_id: str
    similarity_factors: list[SimilarityFactor] = []
    differences: list[Difference] = []
    treatments_observed: list[TreatmentObserved] = []
    confidence: Confidence = "weak"


class ExplainResponse(BaseModel):
    query: str
    explanations: list[CaseExplanation]
    cohort_observation: str = ""
    disclaimer: str
    model_used: str
    degraded: bool = False
    cached: bool = False
    took_ms: int


# ---------------------------------------------------------------- cases

class CaseDetail(BaseModel):
    case_id: str
    document: str
    summary: dict | None
    sex: str
    age: int | None
    age_band: str
    outcome_class: str
    quality_flags: list[str] = []


# ---------------------------------------------------------------- misc

class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    redis: bool
    llm_configured: bool
    points_indexed: int | None = None


class ErrorEnvelope(BaseModel):
    error: str
    detail: str | None = None
    correlation_id: str
