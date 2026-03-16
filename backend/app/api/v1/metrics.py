"""Metrics API — Cache hit ratios and embedding cost metrics.

Exposes:
    GET /v1/metrics/cache      → Redis-based cache hit ratios
    GET /v1/metrics/embedding  → Embedding cost tracker metrics

Note: Graph observability has moved to GET /v1/trace/{trace_id}.
Per-node timing is stored via TraceLogger (Redis) per invocation.

Security note:
    All metrics endpoints are UNAUTHENTICATED (dev-only scope).
    Production deployment MUST add auth guard before exposing publicly.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.services.cache import get_cache_metrics
from app.services.embedding_cost_tracker import get_embedding_metrics

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/cache")
async def cache_metrics(request: Request) -> dict:
    """Return cache hit ratios from Redis counters.

    Process-safe — works across Uvicorn workers and Celery.
    """
    redis = request.app.state.redis
    return await get_cache_metrics(redis)


@router.get("/embedding")
async def embedding_metrics(request: Request) -> dict:
    """Return embedding cost tracker metrics from Redis.

    Includes: total_generated, total_skipped, skip_rate, avg_latency_ms,
    estimated_cost_usd, version_recomputes.

    All counters are atomic (Redis INCR) — safe across Celery workers.
    """
    redis = request.app.state.redis
    return await get_embedding_metrics(redis)
