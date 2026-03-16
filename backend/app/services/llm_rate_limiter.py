"""LLM Rate Limiter.

Per-session token bucket that caps the number of LLM calls per graph
invocation. Prevents runaway costs from retry loops or malformed graphs.

Default limits (configurable):
    max_llm_calls_per_invocation: 10
    max_llm_calls_per_session_minute: 20

Usage:
    limiter = LLMRateLimiter(redis_client)
    if not await limiter.check_and_decrement(session_id, invocation_id):
        raise RuntimeError("LLM rate limit exceeded")
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_CALLS_PER_INVOCATION = 10
_INVOCATION_TTL_SECONDS = 120  # 2 minutes per graph invocation


class LLMRateLimiter:
    """Per-session invocation-scoped rate limiter backed by Redis."""

    def __init__(
        self,
        redis_client: Any,
        *,
        max_calls: int = _MAX_CALLS_PER_INVOCATION,
    ) -> None:
        self._redis = redis_client
        self._max_calls = max_calls

    def _key(self, session_id: str, invocation_id: str) -> str:
        return f"ratelimit:{session_id}:{invocation_id}"

    async def check_and_decrement(
        self,
        session_id: str,
        invocation_id: str,
    ) -> bool:
        """Check if an LLM call is allowed and decrement the counter.

        Returns True if allowed, False if limit exceeded.
        Initializes counter on first call (with TTL).
        """
        key = self._key(session_id, invocation_id)
        try:
            current = await self._redis.get(key)

            if current is None:
                # First call this invocation — initialize counter
                await self._redis.set(key, self._max_calls - 1, ex=_INVOCATION_TTL_SECONDS)
                return True

            remaining = int(current)
            if remaining <= 0:
                logger.warning(
                    "llm_rate_limit_exceeded",
                    session_id=session_id,
                    invocation_id=invocation_id,
                )
                return False

            await self._redis.decr(key)
            return True

        except Exception as e:
            # Redis failures should not block LLM calls — fail open
            logger.error("rate_limiter_error", error=str(e))
            return True

    async def get_remaining(self, session_id: str, invocation_id: str) -> int:
        """Return remaining allowed calls for this invocation."""
        key = self._key(session_id, invocation_id)
        try:
            current = await self._redis.get(key)
            return int(current) if current is not None else self._max_calls
        except Exception:
            return self._max_calls
