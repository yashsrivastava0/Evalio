"""Semantic Cache Service — exact hash match, no cosine at cache layer.

Phase 6 Cache Key: semantic:<embedding_type>:<sha256(embedding_hash:filter_hash)[:24]>
Freshness Check:
    1. Age check: (now - cached_at) > TTL → stale → bypass
    2. Product existence: all cached product_ids must still exist in DB
    3. Availability: none can be unavailable (availability_bool=False)
    4. Price freshness: price not older than PRICE_STALE_AFTER_HOURS

Phase 6 Embed Lock Key: embed:<product_id>:<embedding_type>
    Allows specs and review embeddings to run concurrently per product.

Also manages:
    - Exact query cache: query_cache:<query_hash>
    - Fetch status: fetch_status:<brand>
    - Cache invalidation on price updates / embedding_version bump
    - Redis-based metrics counters (process-safe)
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import orjson

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Redis Metrics Keys ──

METRICS_EXACT_QUERY_HITS = "metrics:exact_query_hits"
METRICS_SEMANTIC_HITS = "metrics:semantic_hits"
METRICS_DB_FAST_PATH_HITS = "metrics:db_fast_path_hits"
METRICS_FETCH_TRIGGER_COUNT = "metrics:fetch_trigger_count"
METRICS_TOTAL_REQUESTS = "metrics:total_requests"

_ALL_METRICS_KEYS = [
    METRICS_EXACT_QUERY_HITS,
    METRICS_SEMANTIC_HITS,
    METRICS_DB_FAST_PATH_HITS,
    METRICS_FETCH_TRIGGER_COUNT,
    METRICS_TOTAL_REQUESTS,
]


async def increment_cache_metric(redis_client: Any, key: str, amount: int = 1) -> None:
    """Atomically increment a Redis metrics counter."""
    try:
        await redis_client.incr(key, amount)
    except Exception as e:
        logger.warning("metrics_incr_failed", key=key, error=str(e))


async def get_cache_metrics(redis_client: Any) -> dict[str, Any]:
    """Retrieve all cache metrics from Redis and compute ratios."""
    try:
        values = await redis_client.mget(*_ALL_METRICS_KEYS)
    except Exception:
        values = [None] * len(_ALL_METRICS_KEYS)

    exact_hits = int(values[0] or 0)
    semantic_hits = int(values[1] or 0)
    fast_path_hits = int(values[2] or 0)
    fetch_triggers = int(values[3] or 0)
    total = int(values[4] or 0)

    return {
        "exact_hit_ratio": round(exact_hits / total, 4) if total > 0 else 0.0,
        "semantic_hit_ratio": round(semantic_hits / total, 4) if total > 0 else 0.0,
        "fast_path_ratio": round(fast_path_hits / total, 4) if total > 0 else 0.0,
        "fetch_ratio": round(fetch_triggers / total, 4) if total > 0 else 0.0,
        "raw": {
            "exact_query_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "db_fast_path_hits": fast_path_hits,
            "fetch_trigger_count": fetch_triggers,
            "total_requests": total,
        },
    }


async def reset_cache_metrics(redis_client: Any) -> None:
    """Reset all cache metrics counters to zero."""
    try:
        await redis_client.delete(*_ALL_METRICS_KEYS)
    except Exception as e:
        logger.warning("metrics_reset_failed", error=str(e))


def build_semantic_cache_key(
    embedding_hash: str,
    filter_hash: str,
    embedding_type: str = "specs",
) -> str:
    """Compute semantic cache key:
        semantic:<embedding_type>:v<EMBEDDING_VERSION>:<sha256(emb_hash:filter_hash:version)[:24]>

    Fix-8: EMBEDDING_VERSION is baked into both the key prefix AND the hash input.
    This guarantees:
        1. Version bump → all keys change → stale cache entries naturally expire.
        2. Two different embedding models at same dimension cannot collide.
        3. A version bump produces a new key namespace without manual flush.

    Still exact hash — no cosine similarity at cache layer.
    """
    from app.core.constants import EMBEDDING_VERSION
    combined = f"{embedding_hash}:{filter_hash}:v{EMBEDDING_VERSION}"
    digest = hashlib.sha256(combined.encode()).hexdigest()[:24]
    return f"semantic:{embedding_type}:v{EMBEDDING_VERSION}:{digest}"


async def get_semantic_cache(
    redis_client: Any,
    embedding_hash: str,
    filter_hash: str,
    *,
    embedding_type: str = "specs",
    max_age_hours: int | None = None,
    db_session: Any = None,   # Optional: for Phase 6 product freshness check
    budget_max: float | None = None,  # Phase 6 fix-B: budget hard filter
) -> dict[str, Any] | None:
    """Look up semantic cache — exact hash match + freshness gate.

    Phase 6 Freshness Gate (in order):
        1. TTL age check (fail → stale)
        2. Product existence + availability check (fail → stale)
        3. Budget boundary check: if budget_max set, any cached product whose
           minimum variant price now exceeds budget_max invalidates the entry.
           This prevents returning products that SQL hard filters would now exclude.

    Returns cached response if fresh, None otherwise.
    """
    settings = get_settings()
    max_age = max_age_hours or settings.price_stale_after_hours

    key = build_semantic_cache_key(embedding_hash, filter_hash, embedding_type)
    raw = await redis_client.get(key)

    if raw is None:
        return None

    try:
        data = orjson.loads(raw)
        cached_at = data.get("cached_at")
        if cached_at:
            cached_dt = datetime.fromisoformat(cached_at)
            age = datetime.now(timezone.utc) - cached_dt
            if age.total_seconds() > max_age * 3600:
                logger.info("semantic_cache_stale", key=key[:32], age_hours=age.total_seconds() / 3600)
                await redis_client.delete(key)
                return None

        # Phase 6: product existence + availability freshness gate
        if db_session is not None:
            cached_ids = data.get("ranked_product_ids", [])
            if cached_ids:
                from sqlalchemy import text as sa_text

                # Gate 2: availability + price recency
                # Note: INTERVAL param must be inlined (PostgreSQL limitation)
                interval_str = f"{int(max_age)} hours"
                check = await db_session.execute(
                    sa_text(f"""
                        SELECT COUNT(DISTINCT p.id) FROM products p
                        JOIN variants v ON v.product_id = p.id
                        WHERE p.id = ANY(:ids)
                          AND v.availability_bool = TRUE
                          AND (
                            v.last_price_update IS NULL
                            OR v.last_price_update > NOW() - INTERVAL '{interval_str}'
                          )
                    """),
                    {"ids": cached_ids},
                )
                live_count = check.scalar_one_or_none() or 0
                if live_count < len(cached_ids):
                    logger.info(
                        "semantic_cache_stale_products",
                        key=key[:32],
                        expected=len(cached_ids),
                        live=live_count,
                    )
                    await redis_client.delete(key)
                    return None

                # Gate 3 (Fix-B): budget drift check
                # If budget_max is set, ensure no cached product's min price
                # now exceeds budget_max (hard filter contract must hold).
                if budget_max is not None:
                    budget_check = await db_session.execute(
                        sa_text("""
                            SELECT COUNT(DISTINCT p.id) FROM products p
                            JOIN variants v ON v.product_id = p.id
                            WHERE p.id = ANY(:ids)
                              AND v.price_inr <= :budget_max
                        """),
                        {"ids": cached_ids, "budget_max": float(budget_max)},
                    )
                    within_budget_count = budget_check.scalar_one_or_none() or 0
                    if within_budget_count < len(cached_ids):
                        logger.info(
                            "semantic_cache_budget_drift",
                            key=key[:32],
                            budget_max=budget_max,
                            within_budget=within_budget_count,
                            cached_total=len(cached_ids),
                        )
                        await redis_client.delete(key)
                        return None

        logger.info("semantic_cache_hit", key=key[:32])
        await increment_cache_metric(redis_client, METRICS_SEMANTIC_HITS)
        return data
    except Exception as e:
        logger.error("semantic_cache_error", error=str(e))
        return None


async def set_semantic_cache(
    redis_client: Any,
    embedding_hash: str,
    filter_hash: str,
    response: dict[str, Any],
    *,
    embedding_type: str = "specs",
    ttl_seconds: int = 3600,
) -> None:
    """Store response in semantic cache with timestamp.

    Phase 6: key includes embedding_type for isolated caching.
    Caller should include 'ranked_product_ids' in response for freshness gate.
    """
    key = build_semantic_cache_key(embedding_hash, filter_hash, embedding_type)
    response["cached_at"] = datetime.now(timezone.utc).isoformat()
    await redis_client.set(key, orjson.dumps(response), ex=ttl_seconds)
    logger.info("semantic_cache_set", key=key[:32])


async def set_query_cache(
    redis_client: Any,
    query_hash: str,
    response: dict[str, Any],
    *,
    ttl_seconds: int = 300,
) -> None:
    """Store response in exact query cache."""
    key = f"query_cache:{query_hash}"
    response["cached_at"] = datetime.now(timezone.utc).isoformat()
    await redis_client.set(key, orjson.dumps(response), ex=ttl_seconds)


async def invalidate_product_cache(
    redis_client: Any,
    product_slug: str,
) -> int:
    """Invalidate all cache entries related to a product.

    Called when scraped data updates prices or reviews.
    Uses pattern scan — use sparingly.
    """
    count = 0
    async for key in redis_client.scan_iter(match="semantic:*"):
        raw = await redis_client.get(key)
        if raw and product_slug.encode() in raw:
            await redis_client.delete(key)
            count += 1
    logger.info("cache_invalidated", product_slug=product_slug, keys_deleted=count)
    return count


async def set_fetch_status(
    redis_client: Any,
    brand: str,
    status: str = "fetching",
    *,
    ttl_seconds: int = 300,
) -> None:
    """Set fetch status for a brand (used by DecisionMatrix)."""
    key = f"fetch_status:{brand}"
    await redis_client.set(key, status, ex=ttl_seconds)


async def get_fetch_status(redis_client: Any, brand: str) -> str | None:
    """Get fetch status for a brand."""
    key = f"fetch_status:{brand}"
    return await redis_client.get(key)


# ── Per-product-slug distributed scrape lock ───────────────────────────────────
# Key: fetch:<product_slug>
# TTL: 600s (10 min) — covers worst-case slow scrape + insert + score compute.
# Uses atomic SET NX EX — only one worker can hold the lock at a time.


async def acquire_scrape_lock(
    redis_client: Any,
    product_slug: str,
    ttl: int = 600,
) -> bool:
    """Atomically acquire the scrape lock for a product slug.

    Uses Redis SET NX EX (atomic test-and-set). Returns True if the lock
    was acquired (caller may proceed). Returns False if another worker
    already holds the lock (caller must exit immediately — no retry).

    Args:
        redis_client: Async Redis client.
        product_slug: Unique product identifier (collision-safe slug).
        ttl: Lock TTL in seconds (default 600 — 10 minutes).

    Returns:
        True if lock acquired, False if already locked.
    """
    key = f"fetch:{product_slug}"
    result = await redis_client.set(key, "locked", nx=True, ex=ttl)
    acquired = result is not None  # SET NX returns None on failure, OK string on success
    logger.info("scrape_lock_attempt", slug=product_slug, acquired=acquired)
    return acquired


async def release_scrape_lock(redis_client: Any, product_slug: str) -> None:
    """Release the scrape lock for a product slug.

    Must be called in both success and failure paths (finally block).
    """
    key = f"fetch:{product_slug}"
    await redis_client.delete(key)
    logger.info("scrape_lock_released", slug=product_slug)


# ── Per-product-slug fetch status ─────────────────────────────────────────────
# Separate from the brand-level set_fetch_status() above.
# Key: fetch_status:<product_slug>  (slug-granular, not brand-granular)


async def set_product_fetch_status(
    redis_client: Any,
    product_slug: str,
    status: str,
    ttl: int = 600,
) -> None:
    """Set fetch status for a specific product slug.

    Valid statuses: "fetching" | "ready" | "failed"

    This is distinct from set_fetch_status(brand) which operates at
    brand granularity for the DecisionMatrix fast-path check.

    Args:
        redis_client: Async Redis client.
        product_slug: Collision-safe product slug.
        status: One of "fetching", "ready", "failed".
        ttl: Key TTL in seconds.
    """
    key = f"fetch_status:{product_slug}"
    await redis_client.set(key, status, ex=ttl)
    logger.info("product_fetch_status_set", slug=product_slug, status=status)


async def get_product_fetch_status(
    redis_client: Any,
    product_slug: str,
) -> str | None:
    """Get slug-level fetch status. Returns None if key not set (pre-scrape)."""
    key = f"fetch_status:{product_slug}"
    return await redis_client.get(key)


# ── Per-product-id embedding lock ─────────────────────────────────────────────
# Key: embed:<product_id>
# Prevents duplicate embedding generation for the same product.
# Shorter TTL (300s) since embedding is faster than a full scrape.


async def acquire_embed_lock(
    redis_client: Any,
    product_id: int,
    embedding_type: str = "specs",
    ttl: int = 300,
) -> bool:
    """Atomically acquire the embedding lock for a (product_id, embedding_type) pair.

    Phase 6: lock key is scoped to embedding_type so that specs and review
    embeddings can run concurrently for the same product without blocking each other.

    Returns True if lock acquired, False if already locked.
    """
    key = f"embed:{product_id}:{embedding_type}"
    result = await redis_client.set(key, "locked", nx=True, ex=ttl)
    acquired = result is not None
    logger.info("embed_lock_attempt", product_id=product_id, embedding_type=embedding_type, acquired=acquired)
    return acquired


async def release_embed_lock(
    redis_client: Any,
    product_id: int,
    embedding_type: str = "specs",
) -> None:
    """Release the embedding lock for a (product_id, embedding_type) pair."""
    key = f"embed:{product_id}:{embedding_type}"
    await redis_client.delete(key)
    logger.info("embed_lock_released", product_id=product_id, embedding_type=embedding_type)
