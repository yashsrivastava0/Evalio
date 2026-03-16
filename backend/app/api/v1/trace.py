"""GET /v1/trace/{trace_id} — Graph observability endpoint.

Returns per-node timing spans for a completed graph invocation.
Spans are stored in Redis by the TraceLogger with a 1-hour TTL.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_redis
from app.services.trace_logger import get_trace
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/trace/{trace_id}")
async def get_trace_spans(
    trace_id: str,
    redis: object = Depends(get_redis),
) -> dict:
    """Return timing spans for a graph invocation.

    Each span contains: node, status, start_time, duration_ms, tokens (if any).
    Returns 404 if the trace has expired (>1 hour) or never existed.
    """
    spans = await get_trace(redis, trace_id)
    if spans is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found or expired")

    logger.info("trace_fetched", trace_id=trace_id, span_count=len(spans))
    return {"trace_id": trace_id, "spans": spans}
