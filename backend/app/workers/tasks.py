"""Celery worker tasks — Phase 8 production-hardened background jobs.

Worker Pool Contract:
    Celery pool = prefork (worker_concurrency=4). Each worker process owns a
    single thread with its own event loop. No eventlet/gevent monkey-patching.
    _run_async() reuses the thread-local event loop (not asyncio.run()) — safe
    because each prefork process executes tasks serially per thread. No nested
    event loop calls are possible. Loop is created once per process and never
    shared across threads.

    UNSUPPORTED: eventlet, gevent. Switching pool type silently breaks _run_async().

Tasks (canonical Celery names):
    tasks.scrape_product         → scrape_queue
    tasks.normalize_product_task → normalize_queue
    tasks.embed_product_task     → embed_queue
    tasks.refresh_stale_products → refresh_queue

Retry Policy:
    IOError / ConnectionError / TimeoutError → autoretry, max 3, exponential backoff + jitter
    json.JSONDecodeError / pydantic.ValidationError (normalize task only) → no retry, dead-letter log
    LLM Timeout (normalize task only) → 1 retry, then dead-letter log

Lifecycle Tracking:
    fetch_status:<slug> and embed_status:<product_id>:<type> are Redis-only.
    Redis restart loses in-flight status; tasks re-trigger but are idempotent (safe).
    Durable DB lifecycle columns deferred to Phase 9 if operationally required.

Queue cap (backpressure):
    scrape_product rejects new tasks when active task count ≥ MAX_PENDING_SCRAPES.

Structured logging fields on every task:
    task_name, product_id, duration_ms, status, retry_count, request_id
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from celery import Celery
from celery.schedules import crontab
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_trace_id(task_self) -> str:
    """Extract trace_id from Celery task headers, or generate a new one.

    Phase 9: trace_id propagated from graph via headers={'trace_id': ...}.
    If no trace_id in headers (e.g., manual invocation), generate uuid4.
    """
    headers = getattr(task_self.request, 'headers', None) or {}
    return headers.get('trace_id', str(uuid4()))

# ── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "evalio_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,          # 10 min hard limit per scrape
    task_soft_time_limit=540,     # warn at 9 min
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    # ── Queue routing ──────────────────────────────────────────────────────────
    task_routes={
        "tasks.scrape_product":          {"queue": "scrape_queue"},
        "tasks.normalize_product_task":  {"queue": "normalize_queue"},
        "tasks.embed_product_task":      {"queue": "embed_queue"},
        "tasks.refresh_stale_products":  {"queue": "refresh_queue"},
    },
    # ── Beat schedule ──────────────────────────────────────────────────────────
    beat_schedule={
        "refresh-stale-products-every-6h": {
            "task": "tasks.refresh_stale_products",
            "schedule": crontab(minute=0, hour="*/6"),
        }
    },
)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_PENDING_SCRAPES: int = 10
REFRESH_BATCH_SIZE: int = 100   # max products per Beat invocation
LLM_TIMEOUT_RETRIES: int = 1    # LLM timeout → 1 retry, then dead-letter


# ── Backpressure guard ─────────────────────────────────────────────────────────


def _active_scrape_count() -> int:
    """Return number of currently active scrape_product tasks across all workers.

    Uses Celery inspector (Redis-backed) to query active task counts.
    Falls back to 0 on any inspection failure to avoid false-positive rejection.
    """
    try:
        inspector = celery_app.control.inspect(timeout=1.0)
        active: dict | None = inspector.active()
        if not active:
            return 0
        return sum(
            len([t for t in tasks if "scrape_product" in t.get("name", "")])
            for tasks in active.values()
        )
    except Exception:
        logger.warning("celery_inspector_error — defaulting active_count=0")
        return 0


# ── Event loop helper ──────────────────────────────────────────────────────────


def _run_async(coro):
    """Run an async coroutine in a per-process thread-local event loop.

    Worker Safety Contract:
        - Celery pool = prefork (default). Each worker process = one thread = one loop.
        - Reuses the thread-local event loop (not asyncio.run()), which avoids
          creating and closing a new loop on every task call. Safe because
          each prefork process executes tasks serially per thread.
        - MUST NOT be called from inside a running coroutine (no nested loops).
        - MUST NOT be used with eventlet/gevent pools (unsupported).
        - Loop is created once per process on first task invocation, then reused.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── Tasks ──────────────────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="tasks.scrape_product",
    max_retries=3,
    autoretry_for=(IOError, ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
)
def scrape_product(
    self,
    product_slug: str,
    category: str,
    url: str | None = None,
    vendor: str | None = None,
    product_name: str | None = None,
) -> dict:
    """Two-tier scrape: Tier-1 (Smartprix/91mobiles) → Tier-2 (discovery) → normalize → upsert.

    Canonical Celery task name: tasks.scrape_product
    Dispatched ONLY by _make_fetch_trigger_node() in graph.py (non-blocking .delay()).
    Never call .get() or .wait() from the graph — fire-and-forget only.

    Backpressure: rejects if active scrape count ≥ MAX_PENDING_SCRAPES.

    Args:
        product_slug: Pre-computed collision-safe slug.
        category: Canonical category ("smartphone" | "laptop"). Never empty.
        url: Optional URL hint for the legacy / refresh path.
        vendor: Optional vendor hint for legacy compatibility.
        product_name: Human-readable name for Tier-1/2 discovery search.

    Returns:
        Result dict: {status, product_id, product_slug, tier, *_ms timings}.
    """
    t_start = time.monotonic()
    retry_count = self.request.retries
    trace_id = _get_trace_id(self)

    if not category:
        raise ValueError("category must be supplied by the caller — never empty")

    logger.info(
        "scrape_task_start",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.scrape_product",
            "product_slug": product_slug,
            "vendor": vendor or "discovery",
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )

    # ── Queue cap / backpressure guard ─────────────────────────────────────────
    active = _active_scrape_count()
    if active >= MAX_PENDING_SCRAPES:
        logger.warning(
            "scrape_task_queue_cap_hit",
            extra={"active": active, "cap": MAX_PENDING_SCRAPES, "slug": product_slug},
        )
        return {
            "status": "rejected",
            "reason": "queue_cap",
            "active_scrapes": active,
            "cap": MAX_PENDING_SCRAPES,
            "product_slug": product_slug,
        }

    # ── Wire to full async two-tier fetch pipeline ─────────────────────────────
    from scraper.workers.fetch_task import run_fetch_task
    from app.db.session import get_async_session
    from app.core.redis_client import get_redis_client

    async def _run():
        async with get_async_session() as db_session:
            redis = await get_redis_client()
            result = await run_fetch_task(
                product_slug=product_slug,
                category=category,
                db_session=db_session,
                redis_client=redis,
                url=url,
                vendor=vendor,
                product_name=product_name,
            )
            # Trigger normalize pipeline on success ("ready" is the success status)
            product_id = result.get("product_id")
            if product_id and result.get("status") == "ready":
                normalize_product_task.apply_async(
                    args=[product_id],
                    queue="normalize_queue",
                )
                logger.info(
                    "scrape_triggered_normalize",
                    extra={
                        "task_name": "tasks.scrape_product",
                        "product_id": product_id,
                        "product_slug": product_slug,
                        "tier": result.get("tier", "unknown"),
                    },
                )
            return result

    result = _run_async(_run())
    duration_ms = int((time.monotonic() - t_start) * 1000)

    logger.info(
        "scrape_task_end",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.scrape_product",
            "product_slug": product_slug,
            "status": result.get("status"),
            "duration_ms": duration_ms,
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )
    return result


@celery_app.task(
    bind=True,
    name="tasks.normalize_product_task",
    max_retries=3,
    autoretry_for=(IOError, ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
)
def normalize_product_task(self, product_id: int) -> dict:
    """Normalize product: Pydantic extraction → LLM at T=0 → set normalize_status.

    LLM Failure Routing:
        - json.JSONDecodeError / pydantic.ValidationError → normalize_status=failed,
          NO retry. Structural failure means retrying will produce the same bad output.
        - LLM timeout → catch, retry once (LLM_TIMEOUT_RETRIES=1), then dead-letter log.
        - IOError / ConnectionError / TimeoutError → standard 3-retry with backoff.

    On success: triggers embed_product_task for both "specs" and "review" types.

    Args:
        product_id: DB product ID to normalize.
    """
    t_start = time.monotonic()
    retry_count = self.request.retries
    trace_id = _get_trace_id(self)

    logger.info(
        "normalize_task_start",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.normalize_product_task",
            "product_id": product_id,
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )

    from app.db.session import get_async_session
    from app.core.redis_client import get_redis_client

    async def _run():
        async with get_async_session() as db_session:
            redis = await get_redis_client()
            normalize_status_key = f"normalize_status:{product_id}"

            try:
                from scraper.normalization.llm_normalizer import normalize_product
                from pydantic import ValidationError

                normalized = await normalize_product(
                    product_id=product_id,
                    db_session=db_session,
                )
                await redis.set(normalize_status_key, "ready", ex=3600)
                logger.info(
                    "normalize_task_success",
                    extra={
                        "task_name": "tasks.normalize_product_task",
                        "product_id": product_id,
                    },
                )
                return {"status": "ok", "product_id": product_id, "normalized": normalized}

            except (json.JSONDecodeError, ValueError) as exc:
                # Structural LLM output failure — no retry
                await redis.set(normalize_status_key, "failed", ex=3600)
                logger.error(
                    "normalize_task_json_failed",
                    extra={
                        "task_name": "tasks.normalize_product_task",
                        "product_id": product_id,
                        "error": str(exc),
                        "retry_count": retry_count,
                    },
                )
                return {"status": "failed", "reason": "json_decode_error", "product_id": product_id}

            except Exception as exc:
                # Check for LLM timeout specifically
                exc_name = type(exc).__name__
                if "timeout" in exc_name.lower() or "timeout" in str(exc).lower():
                    if retry_count < LLM_TIMEOUT_RETRIES:
                        logger.warning(
                            "normalize_task_llm_timeout_retry",
                            extra={
                                "task_name": "tasks.normalize_product_task",
                                "product_id": product_id,
                                "retry_count": retry_count,
                            },
                        )
                        await redis.set(normalize_status_key, "retrying", ex=300)
                        raise  # let autoretry handle it
                    else:
                        await redis.set(normalize_status_key, "failed", ex=3600)
                        logger.error(
                            "normalize_task_llm_timeout_deadletter",
                            extra={
                                "task_name": "tasks.normalize_product_task",
                                "product_id": product_id,
                                "retry_count": retry_count,
                                "error": str(exc),
                            },
                        )
                        return {
                            "status": "failed",
                            "reason": "llm_timeout_exhausted",
                            "product_id": product_id,
                        }
                await redis.set(normalize_status_key, "failed", ex=3600)
                raise  # re-raise for autoretry

    result = _run_async(_run())

    # Trigger embed on success
    if result.get("status") == "ok":
        for embed_type in ("specs", "review"):
            embed_product_task.apply_async(
                args=[product_id, embed_type],
                queue="embed_queue",
            )
        logger.info(
            "normalize_triggered_embed",
            extra={
                "task_name": "tasks.normalize_product_task",
                "product_id": product_id,
            },
        )

    duration_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "normalize_task_end",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.normalize_product_task",
            "product_id": product_id,
            "status": result.get("status"),
            "duration_ms": duration_ms,
            "cost_usd": result.get("cost_usd", 0.0),
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )
    return result


@celery_app.task(
    bind=True,
    name="tasks.embed_product_task",
    max_retries=3,
    autoretry_for=(IOError, ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
)
def embed_product_task(self, product_id: int, embedding_type: str = "specs") -> dict:
    """Generate and upsert embedding for a product.

    Lock: embed:<product_id>:<embedding_type> (SET NX, TTL=300s).
    Idempotent: second caller with same key exits immediately (lock_held).

    Embedding Upsert (compare-and-set):
        Reads product's current updated_at at task start (captured_at).
        Uses atomic INSERT ... ON CONFLICT DO UPDATE ... WHERE updated_at <= :captured_at.
        If a concurrent task wrote a newer embedding, the guard fails silently (no-op).
        This is defense-in-depth — the Redis lock is the primary race guard.

    Args:
        product_id: DB product ID to embed.
        embedding_type: "specs" or "review". Each type is locked independently.
    """
    t_start = time.monotonic()
    retry_count = self.request.retries
    trace_id = _get_trace_id(self)

    logger.info(
        "embed_task_start",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.embed_product_task",
            "product_id": product_id,
            "embedding_type": embedding_type,
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )

    from app.db.session import get_async_session
    from app.core.redis_client import get_redis_client
    from app.workers.redis_lock import RedisLock

    async def _run():
        async with get_async_session() as db_session:
            redis = await get_redis_client()
            lock_key = f"embed:{product_id}:{embedding_type}"
            embed_status_key = f"embed_status:{product_id}:{embedding_type}"

            async with RedisLock(
                redis,
                key=lock_key,
                ttl=300,
                log_context={"product_id": product_id, "embedding_type": embedding_type},
            ) as acquired:
                if not acquired:
                    logger.info(
                        "embed_task_lock_held",
                        extra={
                            "task_name": "tasks.embed_product_task",
                            "product_id": product_id,
                            "embedding_type": embedding_type,
                        },
                    )
                    return {"status": "skipped", "reason": "lock_held", "product_id": product_id}

                await redis.set(embed_status_key, "embedding", ex=600)

                # Read current updated_at before embedding (for compare-and-set guard)
                from sqlalchemy import text as sa_text
                row = await db_session.execute(
                    sa_text(
                        "SELECT updated_at FROM product_embeddings "
                        "WHERE product_id = :pid AND embedding_type = :etype"
                    ),
                    {"pid": product_id, "etype": embedding_type},
                )
                captured_at_row = row.fetchone()
                captured_at = captured_at_row[0] if captured_at_row else None

                # Generate embedding
                from app.services.embedding import generate_product_embedding
                embedding_result = await generate_product_embedding(
                    product_id=product_id,
                    embedding_type=embedding_type,
                    db_session=db_session,
                )
                vector = embedding_result["vector"]
                new_hash = embedding_result["embedding_hash"]

                # Compare-and-set upsert:
                # Guards against a slower concurrent task overwriting a newer embedding.
                # If captured_at IS NULL → first embed ever, INSERT always wins.
                # If captured_at IS NOT NULL → only update if no newer row was written.
                await db_session.execute(
                    sa_text("""
                        INSERT INTO product_embeddings
                            (product_id, embedding_type, vector, embedding_hash, updated_at)
                        VALUES
                            (:pid, :etype, :vec, :hash, NOW())
                        ON CONFLICT (product_id, embedding_type) DO UPDATE SET
                            vector         = EXCLUDED.vector,
                            embedding_hash = EXCLUDED.embedding_hash,
                            updated_at     = NOW()
                        WHERE product_embeddings.updated_at <= :captured_at
                           OR product_embeddings.updated_at IS NULL
                    """),
                    {
                        "pid": product_id,
                        "etype": embedding_type,
                        "vec": vector,
                        "hash": new_hash,
                        "captured_at": captured_at,
                    },
                )
                await db_session.commit()

                await redis.set(embed_status_key, "ready", ex=3600)
                logger.info(
                    "embed_task_success",
                    extra={
                        "task_name": "tasks.embed_product_task",
                        "product_id": product_id,
                        "embedding_type": embedding_type,
                        "embedding_hash": new_hash,
                    },
                )
                return {
                    "status": "ok",
                    "product_id": product_id,
                    "embedding_type": embedding_type,
                    "embedding_hash": new_hash,
                }

    result = _run_async(_run())
    duration_ms = int((time.monotonic() - t_start) * 1000)

    logger.info(
        "embed_task_end",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.embed_product_task",
            "product_id": product_id,
            "embedding_type": embedding_type,
            "status": result.get("status"),
            "duration_ms": duration_ms,
            "lock_acquired": result.get("status") != "skipped",
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )
    return result


@celery_app.task(
    bind=True,
    name="tasks.refresh_stale_products",
    max_retries=3,
    autoretry_for=(IOError, ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
)
def refresh_stale_products_task(self) -> dict:
    """Refresh up to REFRESH_BATCH_SIZE TTL-stale products per Beat invocation.

    Queries DB for products where last_scraped_at < NOW() - STALE_THRESHOLD.
    Enqueues scrape_product.apply_async(countdown=i) with 1s spacing per task.
    This prevents thundering-herd spikes against vendor sites after each Beat tick.

    Throttle:
        Max REFRESH_BATCH_SIZE=100 products per run.
        Tasks spaced 1s apart via countdown=i.

    Returns:
        {status, enqueued, batch_size}
    """
    t_start = time.monotonic()
    retry_count = self.request.retries
    trace_id = _get_trace_id(self)

    logger.info(
        "refresh_task_start",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.refresh_stale_products",
            "batch_size": REFRESH_BATCH_SIZE,
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )

    from app.db.session import get_async_session

    async def _run():
        async with get_async_session() as db_session:
            from sqlalchemy import text as sa_text
            from app.core.config import get_settings as _settings

            stale_hours = _settings().price_stale_after_hours

            rows = await db_session.execute(
                sa_text("""
                    SELECT
                        p.id,
                        p.product_slug,
                        p.category,
                        p.model_name,
                        v.vendor,
                        v.affiliate_url
                    FROM products p
                    JOIN variants v ON v.product_id = p.id
                    WHERE (
                        p.updated_at IS NULL
                        OR p.updated_at < NOW() - INTERVAL :stale_interval
                    )
                    ORDER BY p.updated_at ASC NULLS FIRST
                    LIMIT :limit
                """),
                {
                    "stale_interval": f"{int(stale_hours)} hours",
                    "limit": REFRESH_BATCH_SIZE,
                },
            )
            stale_products = rows.fetchall()
            return stale_products

    stale_products = _run_async(_run())

    enqueued: int = 0
    for i, row in enumerate(stale_products):
        product_id, product_slug, category, model_name, vendor, affiliate_url = row
        if not category:
            continue
        scrape_product.apply_async(
            kwargs={
                "product_slug": product_slug,
                "category": category,
                "url": affiliate_url,      # optional URL hint for Tier-2 fallback
                "vendor": vendor,
                "product_name": model_name,  # human-readable name for Tier-1 search
            },
            queue="scrape_queue",
            countdown=i,  # 1s spacing between tasks
        )
        enqueued += 1
        logger.info(
            "refresh_enqueued_scrape",
            extra={
                "task_name": "tasks.refresh_stale_products",
                "product_id": product_id,
                "product_slug": product_slug,
                "countdown_s": i,
            },
        )

    duration_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "refresh_task_end",
        extra={
            "trace_id": trace_id,
            "task_name": "tasks.refresh_stale_products",
            "enqueued": enqueued,
            "batch_size": REFRESH_BATCH_SIZE,
            "status": "ok",
            "duration_ms": duration_ms,
            "retry_count": retry_count,
            "request_id": self.request.id,
        },
    )
    return {
        "status": "ok",
        "enqueued": enqueued,
        "batch_size": REFRESH_BATCH_SIZE,
    }


# ── Legacy stub tasks (preserved for backward compat; not on Phase 8 queues) ───


@celery_app.task(name="tasks.refresh_prices")
def refresh_prices(category: str = "all", max_products: int = 50) -> dict:
    """Batch price refresh — legacy stub. Replaced by refresh_stale_products in Phase 8."""
    logger.info("refresh_prices_invoked", extra={"category": category, "max": max_products})
    return {
        "status": "superseded",
        "message": "Use refresh_stale_products_task (Phase 8). This stub is preserved for backward compat.",
        "category": category,
    }


@celery_app.task(name="tasks.calibrate_weights")
def calibrate_weights() -> dict:
    """Weight calibration — deferred to Phase 9 golden dataset setup."""
    logger.info("calibrate_weights_invoked")
    return {
        "status": "pending_phase9",
        "message": "Weight calibration deferred to Phase 9 golden dataset setup",
    }


@celery_app.task(name="tasks.recompute_scores")
def recompute_scores(product_ids: list[int] | None = None) -> dict:
    """Recompute component scores — deferred to Phase 9."""
    from app.domain.scoring_config import SCORE_VERSION
    logger.info(
        "recompute_scores_invoked",
        extra={"product_ids": product_ids, "current_version": SCORE_VERSION},
    )
    return {
        "status": "pending_phase9",
        "requested_ids": product_ids or "all",
        "score_version": SCORE_VERSION,
        "message": "Full recompute batch loop to be wired in Phase 9.",
    }
