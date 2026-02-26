"""app.core.redis_client — Singleton async Redis client for workers and services.

This module provides a process-level singleton Redis client.

Usage:
    from app.core.redis_client import get_redis_client

    redis = await get_redis_client()
    await redis.set("key", "value")

Architectural contract:
    - One client instance per process (singleton pattern).
    - Safe for Celery prefork pool: each worker process creates its own instance.
    - NOT shared across processes (prefork isolation guarantees this).
    - FastAPI's lifespan creates its own client via aioredis.from_url() directly
      (stored in app.state.redis). This module is for Celery workers + non-FastAPI
      consumers that cannot access app.state.
    - decode_responses=True: all values are returned as Python str, not bytes.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.core.config import get_settings

_redis_client: aioredis.Redis | None = None


async def get_redis_client() -> aioredis.Redis:
    """Return (or lazily create) the process-level singleton Redis client.

    Thread-safety note:
        In Celery prefork mode, each worker process is single-threaded for task
        execution. The singleton is created on first call within a process and
        reused for all subsequent calls. No locking is required.

    Returns:
        Async Redis client with decode_responses=True.
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=10,
        )
    return _redis_client


async def close_redis_client() -> None:
    """Close the singleton Redis client (call during process shutdown if needed)."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
