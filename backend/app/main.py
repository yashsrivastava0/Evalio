"""Evalio AI Product Decision Engine — FastAPI application entrypoint.

Wires up:
- Lifespan (DB + Redis + LLM client + embedding provider init/shutdown)
- CORS
- Correlation-ID middleware
- API v1 routers
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.logging import CorrelationIdMiddleware, setup_logging, get_logger
from app.models.database import engine
from app.services.embedding import OpenAIEmbeddingProvider

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize and tear down shared resources."""
    settings = get_settings()
    setup_logging(settings.log_level)

    # ── Redis connection pool ──
    app.state.redis = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        max_connections=20,
    )
    logger.info("redis_connected", url=settings.redis_url)

    # ── LLM client (shared AsyncOpenAI) ──
    app.state.llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
    logger.info("llm_client_initialized", model=settings.llm_model)

    # ── Embedding provider ──
    app.state.embedding_provider = OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    logger.info("embedding_provider_initialized", model=settings.embedding_model)

    # ── Verify DB connection ──
    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )
    logger.info("database_connected", url=settings.database_url[:40] + "...")

    yield

    # ── Shutdown ──
    await app.state.redis.close()
    await engine.dispose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title="Evalio API",
        description="AI Product Decision Engine for Smartphones & Laptops",
        version="2.0.0",
        lifespan=lifespan,
    )

    # ── Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)

    # ── Routers ──
    from app.api.v1.chat import router as chat_router
    from app.api.v1.search import router as search_router
    from app.api.v1.compare import router as compare_router
    from app.api.v1.scrape import router as scrape_router
    from app.api.v1.metrics import router as metrics_router
    from app.api.v1.trace import router as trace_router

    app.include_router(chat_router, prefix="/v1", tags=["chat"])
    app.include_router(search_router, prefix="/v1", tags=["search"])
    app.include_router(compare_router, prefix="/v1", tags=["compare"])
    app.include_router(scrape_router, prefix="/v1", tags=["scrape"])
    app.include_router(metrics_router, prefix="/v1", tags=["metrics"])
    app.include_router(trace_router, prefix="/v1", tags=["observability"])

    # ── Health check ──
    @app.get("/health", tags=["infra"])
    async def health() -> dict[str, str]:
        """Probe each service and return structured health status.

        Always returns HTTP 200. Callers inspect per-service values:
            "connected"    — service is reachable
            "disconnected" — service probe failed

        Does NOT raise on failure — fail-fast is enforced at startup
        (lifespan), not here.
        """
        db_status = "disconnected"
        redis_status = "disconnected"
        celery_status = "disconnected"

        # ── Postgres probe ──
        try:
            import sqlalchemy
            async with engine.begin() as conn:
                await conn.execute(sqlalchemy.text("SELECT 1"))
            db_status = "connected"
        except Exception:
            pass

        # ── Redis probe ──
        try:
            redis_client = app.state.redis
            await redis_client.ping()
            redis_status = "connected"
        except Exception:
            pass

        # ── Celery probe (non-blocking, short timeout) ──
        # Import from app.workers.tasks — that is where celery_app is defined.
        try:
            from app.workers.tasks import celery_app
            response = celery_app.control.ping(timeout=0.5)
            celery_status = "connected" if response else "disconnected"
        except Exception:
            pass

        return {
            "status": "ok",
            "database": db_status,
            "redis": redis_status,
            "celery": celery_status,
        }

    return app


app = create_app()
