"""Application configuration — loaded from environment / .env (spec §19.3)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app
    app_name: str = "MedSearch AI"
    debug: bool = False

    # --- qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "cases_v1"
    embedding_version: str = "bgem3-v1"

    # --- embedding model
    embed_model_name: str = "BAAI/bge-m3"

    # --- llm (NVIDIA NIM, OpenAI-compatible)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model_primary: str = "meta/llama-3.3-70b-instruct"
    llm_model_fallback: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2

    # --- redis (explanation/search cache)
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # --- rerank ("thorough mode", NVIDIA NIM hosted cross-encoder)
    # Measured on the gold set: nDCG@10 0.942, R@1 0.929 (evaluation/RESULTS.md)
    rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    rerank_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
    rerank_depth: int = 50       # retrieval candidates sent to the reranker
    rerank_beta: float = 0.9     # blend: beta*rerank + (1-beta)*retrieval
    rerank_doc_chars: int = 1600 # longer text measurably hurts precision

    # --- negation-aware scoring (NegEx-lite, spec §15.2)
    negation_enabled: bool = True
    negation_penalty: float = 0.25  # per contradicted term, multiplicative
    negation_bonus: float = 0.05    # shared pertinent negative
    negation_pool: int = 30         # fast-mode candidate pool before adjustment

    # --- data
    cases_file: Path = PROJECT_ROOT / "data" / "processed" / "cases_clean.jsonl"

    # --- retrieval defaults
    default_k: int = 10
    max_k: int = 25
    prefetch_limit: int = 100
    # weighted-score fusion: alpha * norm(dense) + (1-alpha) * norm(sparse).
    # alpha=0.4 won the evaluation sweep (evaluation/RESULTS.md):
    # nDCG@10 0.658 vs 0.625 for RRF; Recall@10 0.859 vs 0.848.
    fusion_alpha: float = 0.4
    max_case_context_chars: int = 4500


settings = Settings()
