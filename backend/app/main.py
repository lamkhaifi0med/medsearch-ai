"""FastAPI application factory (spec §19.3).

Startup (lifespan): load BGE-M3 once, build the case index, connect Redis.
Middleware: correlation IDs on every request, uniform error envelopes.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routes import router as v1_router
from app.core.config import settings
from app.core.logging import correlation_id_var, new_correlation_id, setup_logging
from app.services.cases import case_store
from app.services.llm import llm_gateway
from app.services.rerank import rerank_service
from app.services.retrieval import retrieval_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.debug)
    logger.info("Starting %s", settings.app_name)
    case_store.build_index()
    llm_gateway.connect_cache()
    # rerank doc cache: ~50s of bind-mount I/O — warm it off the critical path
    threading.Thread(target=rerank_service.warm, daemon=True).start()
    retrieval_service.load_model()  # slowest step (~30s) — do it last, once
    logger.info("Startup complete — API ready")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Clinical case retrieval with semantic search and grounded RAG explanations. "
                    "Decision support only — never diagnostic.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],  # dev frontend
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        cid = new_correlation_id()
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": None,  # never leak internals
                "correlation_id": correlation_id_var.get(),
            },
        )

    app.include_router(v1_router)
    return app


app = create_app()
