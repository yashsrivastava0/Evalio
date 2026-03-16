"""Graph Observability — Trace Logger.

Records per-node timing, token usage, and errors for each graph invocation.
Stored in Redis as trace:<trace_id> (JSON list of spans) with a 1-hour TTL.
Readable via GET /v1/trace/{trace_id} for debugging.

Usage:
    tracer = TraceLogger(redis_client, trace_id)
    await tracer.node_start("intent")
    ...
    await tracer.node_end("intent", duration_ms=120, tokens=350)
"""

from __future__ import annotations

import time
from typing import Any

import orjson

from app.core.logging import get_logger

logger = get_logger(__name__)

_TRACE_TTL_SECONDS = 3600  # 1 hour


class TraceLogger:
    """Per-request trace logger backed by Redis."""

    def __init__(self, redis_client: Any, trace_id: str) -> None:
        self._redis = redis_client
        self._trace_id = trace_id
        self._key = f"trace:{trace_id}"
        self._timers: dict[str, float] = {}

    async def node_start(self, node: str) -> None:
        """Record node start and store wall-clock start time."""
        self._timers[node] = time.monotonic()
        await self._append_span({
            "node": node,
            "event": "start",
            "ts": time.time(),
        })

    async def node_end(
        self,
        node: str,
        *,
        tokens: int | None = None,
        model: str | None = None,
    ) -> int:
        """Record node end. Returns elapsed milliseconds."""
        start = self._timers.pop(node, None)
        duration_ms = int((time.monotonic() - start) * 1000) if start is not None else 0
        await self._append_span({
            "node": node,
            "event": "end",
            "duration_ms": duration_ms,
            "tokens": tokens,
            "model": model,
            "ts": time.time(),
        })
        return duration_ms

    async def node_error(self, node: str, error: str) -> None:
        """Record a node-level error."""
        logger.error("trace_node_error", trace_id=self._trace_id, node=node, error=error)
        await self._append_span({
            "node": node,
            "event": "error",
            "error": error,
            "ts": time.time(),
        })

    async def _append_span(self, span: dict) -> None:
        """Append a span to the Redis list atomically."""
        try:
            raw = await self._redis.get(self._key)
            spans: list[dict] = orjson.loads(raw) if raw else []
            spans.append(span)
            await self._redis.set(self._key, orjson.dumps(spans), ex=_TRACE_TTL_SECONDS)
        except Exception as e:
            # Tracing failures must never impact the main request
            logger.warning("trace_write_failed", trace_id=self._trace_id, error=str(e))

    async def get_spans(self) -> list[dict]:
        """Fetch all recorded spans for this trace."""
        try:
            raw = await self._redis.get(self._key)
            return orjson.loads(raw) if raw else []
        except Exception:
            return []


async def get_trace(redis_client: Any, trace_id: str) -> list[dict]:
    """Fetch trace spans by trace_id. Used by the /v1/trace endpoint."""
    try:
        raw = await redis_client.get(f"trace:{trace_id}")
        return orjson.loads(raw) if raw else []
    except Exception:
        return []
