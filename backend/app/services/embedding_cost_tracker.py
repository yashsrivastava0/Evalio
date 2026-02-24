"""Embedding Cost Tracker — Redis-based counters for embedding economics.

Tracks:
    - total_generated: Number of embeddings successfully generated
    - total_skipped: Embeddings skipped due to hash dedupe (cost saved)
    - total_latency_ms: Cumulative generation latency in milliseconds
    - version_recomputes: Force-recomputes triggered by embedding_version bump

Exposed via GET /v1/metrics/embedding.

All counters use atomic Redis INCR — same pattern as cache metrics.
Cost is MEASURED from real latency + provider pricing, never guessed.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Redis Counter Keys ─────────────────────────────────────────────────────────

EMB_METRICS_TOTAL_GENERATED = "emb_metrics:total_generated"
EMB_METRICS_TOTAL_SKIPPED = "emb_metrics:total_skipped"
EMB_METRICS_TOTAL_LATENCY_MS = "emb_metrics:total_latency_ms"
EMB_METRICS_VERSION_RECOMPUTES = "emb_metrics:version_recomputes"

_ALL_EMB_METRICS_KEYS = [
    EMB_METRICS_TOTAL_GENERATED,
    EMB_METRICS_TOTAL_SKIPPED,
    EMB_METRICS_TOTAL_LATENCY_MS,
    EMB_METRICS_VERSION_RECOMPUTES,
]

# Provider pricing per 1k embeddings (USD) — update when provider changes rates
_PROVIDER_COST_PER_1K = {
    "openai:text-embedding-3-small": 0.020,
    "openai:text-embedding-3-large": 0.130,
    "gemini:text-embedding-004": 0.000,  # Currently free tier
}


async def increment_embedding_metric(
    redis_client: Any,
    key: str,
    amount: int = 1,
) -> None:
    """Atomically increment a Redis embedding metrics counter."""
    try:
        await redis_client.incr(key, amount)
    except Exception as e:
        logger.warning("emb_metrics_incr_failed", key=key, error=str(e))


async def get_embedding_metrics(redis_client: Any) -> dict[str, Any]:
    """Retrieve all embedding metrics and compute derived stats.

    Returns:
        {
            "total_generated": int,
            "total_skipped": int,
            "avg_latency_ms": float,
            "skip_rate": float,          # skipped / (generated + skipped)
            "estimated_cost_usd": float, # based on measured generation count
            "version_recomputes": int,
            "raw": {...}
        }
    """
    try:
        values = await redis_client.mget(*_ALL_EMB_METRICS_KEYS)
    except Exception:
        values = [None] * len(_ALL_EMB_METRICS_KEYS)

    generated = int(values[0] or 0)
    skipped = int(values[1] or 0)
    total_latency_ms = int(values[2] or 0)
    version_recomputes = int(values[3] or 0)

    total_attempts = generated + skipped
    avg_latency_ms = round(total_latency_ms / generated, 2) if generated > 0 else 0.0
    skip_rate = round(skipped / total_attempts, 4) if total_attempts > 0 else 0.0

    # Cost estimate: using default openai small model pricing
    # Real cost = generated embeddings / 1000 * price_per_1k
    cost_per_1k = _PROVIDER_COST_PER_1K.get("openai:text-embedding-3-small", 0.020)
    estimated_cost_usd = round((generated / 1000) * cost_per_1k, 6)

    return {
        "total_generated": generated,
        "total_skipped": skipped,
        "avg_latency_ms": avg_latency_ms,
        "skip_rate": skip_rate,
        "estimated_cost_usd": estimated_cost_usd,
        "version_recomputes": version_recomputes,
        "raw": {
            "total_generated": generated,
            "total_skipped": skipped,
            "total_latency_ms": total_latency_ms,
            "version_recomputes": version_recomputes,
        },
    }


async def reset_embedding_metrics(redis_client: Any) -> None:
    """Reset all embedding metrics counters to zero. For testing only."""
    try:
        await redis_client.delete(*_ALL_EMB_METRICS_KEYS)
    except Exception as e:
        logger.warning("emb_metrics_reset_failed", error=str(e))
