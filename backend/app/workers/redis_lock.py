"""Redis distributed lock — async context manager for Celery tasks.

Worker Safety Contract:
    - Celery pool = prefork (default, worker_concurrency=4).
    - Each worker process owns a single thread with its own event loop.
    - _run_async() in tasks.py reuses the thread-local event loop (not asyncio.run()).
    - RedisLock must only be used inside coroutines passed to _run_async().
    - Never call RedisLock from a synchronous Celery task body directly.

Usage (correct — inside _run_async coroutine):
    async def _embed_impl(product_id):
        async with RedisLock(
            redis, key=f"embed:{product_id}:specs", ttl=300
        ) as acquired:
            if not acquired:
                logger.info("lock_not_acquired", key=f"embed:{product_id}:specs")
                return {"status": "skipped", "reason": "lock_held"}
            # proceed with work ...

    return _run_async(_embed_impl(product_id))
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisLock:
    """Async context manager wrapping Redis SET NX EX atomic lock.

    Wraps the low-level SET NX / DEL primitives for ergonomic use
    inside Celery task coroutines (passed to _run_async()).

    Args:
        redis_client: Async Redis client.
        key: Full Redis key to lock (e.g. "fetch:<slug>", "embed:<id>:specs").
        ttl: Lock TTL in seconds. Must exceed worst-case task duration.
        log_context: Optional dict of extra fields for structured logs.

    Guarantees:
        - Lock is always released in __aexit__, even on exception.
        - If lock already held (acquired=False), __aexit__ does NOT delete the key.
        - Idempotent: second caller with same key gets acquired=False immediately.
    """

    def __init__(
        self,
        redis_client: Any,
        key: str,
        *,
        ttl: int = 600,
        log_context: dict | None = None,
    ) -> None:
        self._redis = redis_client
        self._key = key
        self._ttl = ttl
        self._log_context = log_context or {}
        self._acquired = False

    async def __aenter__(self) -> bool:
        result = await self._redis.set(self._key, "locked", nx=True, ex=self._ttl)
        self._acquired = result is not None
        if self._acquired:
            logger.info(
                "redis_lock_acquired",
                extra={"key": self._key, "ttl": self._ttl, **self._log_context},
            )
        else:
            logger.info(
                "redis_lock_not_acquired",
                extra={"key": self._key, **self._log_context},
            )
        return self._acquired

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._acquired:
            await self._redis.delete(self._key)
            logger.info(
                "redis_lock_released",
                extra={"key": self._key, **self._log_context},
            )
        # Do not suppress exceptions
        return False
