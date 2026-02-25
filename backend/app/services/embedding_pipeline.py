"""Embedding Pipeline — multi-type embedding generation with hash dedupe.

Phase 6 architectural rules:
    - Always called from Celery tasks — NEVER from user request path.
    - Embed lock scoped per (product_id, embedding_type) — specs and review can
      run concurrently for the same product.
    - specs_hash dedupe: skip re-embed if specs_jsonb is unchanged.
    - reviews_hash dedupe: skip if summary_text is unchanged.
    - Guard: review embedding skipped if summary_text is None or empty.
    - embedding_version bump forces re-embed regardless of hash.
    - Lock always released in finally block.
    - No LLM calls. No ranking math. No cross-node imports.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import EMBEDDING_VERSION  # single source of truth — never redefine here
from app.core.logging import get_logger
from app.models.tables import Embedding, EmbeddingType, Product, ReviewSummary
from app.services.cache import acquire_embed_lock, release_embed_lock

logger = get_logger(__name__)

# Re-export so existing imports of EMBEDDING_VERSION from this module still work.
__all__ = ["EMBEDDING_VERSION", "compute_specs_hash", "compute_reviews_hash",
           "compute_embedding_hash", "should_reembed",
           "generate_specs_embedding", "generate_review_embedding"]


class DimensionMismatchError(RuntimeError):
    """Raised when provider returns a vector with unexpected dimension.

    Fix-5: Catches embedding model dimension changes before they corrupt
    the pgvector column. If EMBEDDING_VERSION is bumped alongside a model
    change, this guard fires immediately rather than silently writing a
    mismatched vector to the database.

    On dimension change: update app/core/constants.py AND run Alembic
    migration to ALTER the embeddings.embedding_vector column type.
    """


# ── Hash Utilities ──────────────────────────────────────────────────────────────


def _canonical_json_value(v: Any) -> Any:
    """Recursively normalize a JSON value for canonical serialization.

    Rules:
        - Dicts: sorted keys, nulls stripped recursively
        - Lists: each element normalized
        - Floats: rounded to 8 decimal places (6.70 == 6.7, no precision drift)
        - Strings, ints, bools: unchanged
        - None: caller strips at dict level; bare None stays None
    """
    if isinstance(v, dict):
        return {
            k: _canonical_json_value(vv)
            for k, vv in sorted(v.items())
            if vv is not None  # strip null fields consistently
        }
    if isinstance(v, list):
        return [_canonical_json_value(item) for item in v]
    if isinstance(v, float):
        return round(v, 8)  # normalize: 6.70 ≡ 6.7, prevents false hash diffs
    return v


def compute_specs_hash(specs_jsonb: dict | None) -> str:
    """Compute deterministic sha256 hash of product specs.

    Canonical JSON guarantees:
        1. Keys sorted recursively (nested dicts too)
        2. Null fields stripped consistently (None fields don't appear)
        3. Floats normalized to 8dp (6.70 and 6.7 produce identical hash)
        4. ASCII encoding, no whitespace

    This is the single source of truth for specs content identity.
    If specs_jsonb changes, hash changes. If specs_jsonb is unchanged, hash is stable.
    Returns 64-char hex string.
    """
    if not specs_jsonb:
        return hashlib.sha256(b"{}").hexdigest()
    canonical_data = _canonical_json_value(specs_jsonb)
    canonical = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_reviews_hash(review_text: str | None) -> str | None:
    """Compute sha256 hash of review summary text.

    Returns None if review_text is None or empty (signals: no review data).
    """
    if not review_text or not review_text.strip():
        return None
    return hashlib.sha256(review_text.strip().encode("utf-8")).hexdigest()


def compute_embedding_hash(vector: list[float]) -> str:
    """Compact hash of embedding vector for cache keying."""
    key_floats = vector[:32] + vector[-32:] if len(vector) > 64 else vector
    raw = "|".join(f"{f:.6f}" for f in key_floats)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ── Dedupe Check ─────────────────────────────────────────────────────────────────


async def should_reembed(
    product_id: int,
    db: AsyncSession,
    embedding_type: str,
    new_content_hash: str | None,
) -> bool:
    """Return True if embedding needs to be regenerated.

    Skip conditions (returns False):
        - existing embedding found
        - AND stored hash == new_content_hash
        - AND stored embedding_version == EMBEDDING_VERSION

    Force re-embed (returns True):
        - no existing embedding
        - hash changed (content changed)
        - version bumped
        - new_content_hash is None (no-data guard — caller decides skip)
    """
    if new_content_hash is None:
        # No content available — caller should skip embedding
        return False

    result = await db.execute(
        select(Embedding).where(
            Embedding.product_id == product_id,
            Embedding.embedding_type == embedding_type,
        )
    )
    existing: Embedding | None = result.scalar_one_or_none()

    if existing is None:
        logger.info("reembed_no_existing", product_id=product_id, embedding_type=embedding_type)
        return True

    stored_hash = existing.specs_hash if embedding_type == "specs" else existing.reviews_hash
    if stored_hash != new_content_hash:
        logger.info(
            "reembed_hash_changed",
            product_id=product_id,
            embedding_type=embedding_type,
        )
        return True

    if existing.embedding_version != EMBEDDING_VERSION:
        logger.info(
            "reembed_version_bump",
            product_id=product_id,
            embedding_type=embedding_type,
            stored_version=existing.embedding_version,
            current_version=EMBEDDING_VERSION,
        )
        return True

    logger.info("dedupe_skip", product_id=product_id, embedding_type=embedding_type)
    return False


# ── Core Pipeline ─────────────────────────────────────────────────────────────────


async def generate_specs_embedding(
    product_id: int,
    db: AsyncSession,
    redis_client: Any,
    provider: Any,
) -> dict[str, Any]:
    """Generate and store specs embedding for a product.

    Pipeline:
        1. Load product.specs_jsonb
        2. Compute specs_hash
        3. Check should_reembed — skip on dedupe
        4. Acquire embed lock (product_id, "specs")
        5. Generate embedding vector
        6. Upsert into embeddings table
        7. Release lock in finally
        8. Track cost metrics

    Returns result dict with status and timing.
    """
    t_start = time.monotonic()

    # Step 1: Load specs
    result = await db.execute(select(Product).where(Product.id == product_id))
    product: Product | None = result.scalar_one_or_none()
    if product is None:
        logger.warning("embed_product_not_found", product_id=product_id)
        return {"status": "error", "reason": "product_not_found"}

    # Step 2: Compute hash
    specs_hash = compute_specs_hash(product.specs_jsonb)

    # Step 3: Dedupe check
    if not await should_reembed(product_id, db, "specs", specs_hash):
        await _track_skip(redis_client)
        return {"status": "skipped", "reason": "dedupe_specs_hash_unchanged"}

    # Step 4: Acquire scoped lock
    lock_acquired = await acquire_embed_lock(redis_client, product_id, "specs")
    if not lock_acquired:
        logger.info("embed_lock_conflict", product_id=product_id, embedding_type="specs")
        return {"status": "skipped", "reason": "lock_conflict"}

    try:
        # Step 5: Generate embedding — build text from specs
        specs_text = _specs_to_text(product.specs_jsonb, product.canonical_title)
        vector: list[float] = await provider.embed(specs_text)

        # Step 5a: Dimension guard (Fix-5) — must match before writing to DB
        _validate_dimension(vector, product_id, "specs")

        embedding_hash = compute_embedding_hash(vector)

        # Step 6: Upsert
        await _upsert_embedding(
            db=db,
            product_id=product_id,
            vector=vector,
            embedding_type="specs",
            embedding_hash=embedding_hash,
            specs_hash=specs_hash,
            reviews_hash=None,
        )
        await db.commit()

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        await _track_generated(redis_client, elapsed_ms)
        logger.info(
            "specs_embedding_generated",
            product_id=product_id,
            latency_ms=elapsed_ms,
        )
        return {"status": "generated", "product_id": product_id, "latency_ms": elapsed_ms}

    except Exception as e:
        logger.error("specs_embedding_error", product_id=product_id, error=str(e))
        await db.rollback()
        return {"status": "error", "reason": str(e)}
    finally:
        await release_embed_lock(redis_client, product_id, "specs")


async def generate_review_embedding(
    product_id: int,
    db: AsyncSession,
    redis_client: Any,
    provider: Any,
) -> dict[str, Any]:
    """Generate and store review embedding for a product.

    Guard: skips if review_summary.summary_text is None or empty.
    Lock scoped to (product_id, 'review') — does NOT block specs embedding.
    """
    t_start = time.monotonic()

    # Load review summary
    result = await db.execute(
        select(ReviewSummary).where(ReviewSummary.product_id == product_id)
    )
    review: ReviewSummary | None = result.scalar_one_or_none()

    # Guard: skip if no review data
    if review is None or not review.summary_text or not review.summary_text.strip():
        logger.info("review_embed_skipped_no_summary", product_id=product_id)
        return {"status": "skipped", "reason": "no_review_summary"}

    # Compute reviews_hash
    reviews_hash = compute_reviews_hash(review.summary_text)
    if reviews_hash is None:
        return {"status": "skipped", "reason": "empty_review_text"}

    # Dedupe check
    if not await should_reembed(product_id, db, "review", reviews_hash):
        await _track_skip(redis_client)
        return {"status": "skipped", "reason": "dedupe_reviews_hash_unchanged"}

    # Acquire scoped lock — "review" type, doesn't conflict with "specs"
    lock_acquired = await acquire_embed_lock(redis_client, product_id, "review")
    if not lock_acquired:
        logger.info("embed_lock_conflict", product_id=product_id, embedding_type="review")
        return {"status": "skipped", "reason": "lock_conflict"}

    try:
        # Generate embedding from review text
        vector: list[float] = await provider.embed(review.summary_text)

        # Dimension guard (Fix-5)
        _validate_dimension(vector, product_id, "review")

        embedding_hash = compute_embedding_hash(vector)

        # Upsert
        await _upsert_embedding(
            db=db,
            product_id=product_id,
            vector=vector,
            embedding_type="review",
            embedding_hash=embedding_hash,
            specs_hash=None,
            reviews_hash=reviews_hash,
        )
        await db.commit()

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        await _track_generated(redis_client, elapsed_ms)
        logger.info(
            "review_embedding_generated",
            product_id=product_id,
            latency_ms=elapsed_ms,
        )
        return {"status": "generated", "product_id": product_id, "latency_ms": elapsed_ms}

    except Exception as e:
        logger.error("review_embedding_error", product_id=product_id, error=str(e))
        await db.rollback()
        return {"status": "error", "reason": str(e)}
    finally:
        await release_embed_lock(redis_client, product_id, "review")


def _validate_dimension(vector: list[float], product_id: int, embedding_type: str) -> None:
    """Validate embedding vector dimension matches expected size.

    Fix-5: If embedding provider dimension changes (e.g., 1536 \u2192 3072),
    pgvector will reject the INSERT silently or corrupt the column.
    This guard fires BEFORE the DB write so the error is explicit.

    Expected dimension is read from settings.embedding_dim (default 1536).
    On model change:
        1. Update settings.embedding_dim
        2. Bump EMBEDDING_VERSION in app/core/constants.py
        3. Run Alembic migration to ALTER embeddings.embedding_vector TYPE
    """
    settings = get_settings()
    expected_dim = getattr(settings, "embedding_dim", 1536)
    actual_dim = len(vector)
    if actual_dim != expected_dim:
        raise DimensionMismatchError(
            f"Embedding dimension mismatch for product_id={product_id} "
            f"type={embedding_type}: expected {expected_dim}, got {actual_dim}. "
            f"If you changed embedding models, update settings.embedding_dim "
            f"AND run Alembic migration to ALTER embeddings.embedding_vector."
        )


def _specs_to_text(specs_jsonb: dict | None, canonical_title: str = "") -> str:
    """Convert specs dict to a flat text string for embedding.

    Joins key-value pairs: "ram_gb: 8 | storage_gb: 128 | ..."
    Prepends canonical_title for better semantic context.
    """
    if not specs_jsonb:
        return canonical_title or ""
    parts = [f"{k}: {v}" for k, v in sorted(specs_jsonb.items()) if v is not None]
    spec_text = " | ".join(parts)
    return f"{canonical_title} | {spec_text}" if canonical_title else spec_text


async def _upsert_embedding(
    db: AsyncSession,
    product_id: int,
    vector: list[float],
    embedding_type: str,
    embedding_hash: str,
    specs_hash: str | None,
    reviews_hash: str | None,
) -> None:
    """Upsert embedding into embeddings table.

    Uses raw SQL ON CONFLICT for atomicity. The unique constraint
    uq_embeddings_product_type (product_id, embedding_type) ensures
    only one embedding per type per product.
    """
    vector_str = "[" + ",".join(str(f) for f in vector) + "]"
    await db.execute(
        text("""
            INSERT INTO embeddings
                (product_id, embedding_vector, embedding_type, embedding_hash,
                 embedding_version, specs_hash, reviews_hash)
            VALUES
                (:product_id, :vector, :emb_type, :emb_hash,
                 :emb_version, :specs_hash, :reviews_hash)
            ON CONFLICT (product_id, embedding_type) DO UPDATE SET
                embedding_vector = EXCLUDED.embedding_vector,
                embedding_hash   = EXCLUDED.embedding_hash,
                embedding_version = EXCLUDED.embedding_version,
                specs_hash       = EXCLUDED.specs_hash,
                reviews_hash     = EXCLUDED.reviews_hash,
                created_at       = NOW()
        """),
        {
            "product_id": product_id,
            "vector": vector_str,
            "emb_type": embedding_type,
            "emb_hash": embedding_hash,
            "emb_version": EMBEDDING_VERSION,
            "specs_hash": specs_hash,
            "reviews_hash": reviews_hash,
        },
    )


async def _track_generated(redis_client: Any, latency_ms: int) -> None:
    """Increment cost tracker counters for a generated embedding."""
    try:
        from app.services.embedding_cost_tracker import (
            EMB_METRICS_TOTAL_GENERATED,
            EMB_METRICS_TOTAL_LATENCY_MS,
        )
        await redis_client.incr(EMB_METRICS_TOTAL_GENERATED)
        await redis_client.incr(EMB_METRICS_TOTAL_LATENCY_MS, latency_ms)
    except Exception:
        pass  # Cost tracking is non-critical


async def _track_skip(redis_client: Any) -> None:
    """Increment cost tracker counter for a skipped (deduped) embedding."""
    try:
        from app.services.embedding_cost_tracker import EMB_METRICS_TOTAL_SKIPPED
        await redis_client.incr(EMB_METRICS_TOTAL_SKIPPED)
    except Exception:
        pass
