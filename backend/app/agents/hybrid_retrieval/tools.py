"""Hybrid Retrieval Agent — Tool functions.

Contains the three core retrieval functions moved verbatim from services/retrieval.py,
plus new tools: cache check, session bypass, improved composite reranking.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.tables import Product, Variant, ComponentScore, Embedding, EmbeddingType
from app.services.entity_resolver import resolve_product_entities, resolve_to_product_ids

logger = get_logger(__name__)


# ── Embedding type selection ───────────────────────────────────────────────────

_REVIEW_KEYWORDS: frozenset[str] = frozenset({
    "camera", "battery life", "battery", "heating", "heat", "thermal",
    "review", "reviews", "durability", "build quality", "real world",
    "overheat", "long term",
})


def select_embedding_type(focus_attributes: list[str], intent_type: str) -> str:
    """Deterministically select embedding type from focus attributes."""
    focus_set = {f.lower() for f in focus_attributes}
    if focus_set & _REVIEW_KEYWORDS:
        return "review"
    if intent_type in ("FEATURE_QUERY",):
        return "review"
    return "specs"


# ── SQL filter ────────────────────────────────────────────────────────────────

async def sql_filter_candidates(
    db: AsyncSession,
    *,
    category: str | None = None,
    budget_min: float | None = None,
    budget_max: float | None = None,
    brands: list[str] | None = None,
    min_ram: int | None = None,
    min_storage: int | None = None,
    availability_only: bool = True,
    max_candidates: int | None = None,
) -> list[int]:
    """Step 1: SQL parameterized filter → reduce to ~300 candidate product IDs."""
    settings = get_settings()
    max_candidates = max_candidates or settings.max_sql_candidates

    conditions = []
    if category:
        conditions.append(Product.category == category)
    if availability_only:
        conditions.append(Variant.availability_bool.is_(True))
    if budget_min is not None:
        conditions.append(Variant.price_inr >= budget_min)
    if budget_max is not None:
        conditions.append(Variant.price_inr <= budget_max)
    if brands:
        conditions.append(Product.brand.in_([b.lower() for b in brands]))
    if min_ram is not None:
        conditions.append(Variant.ram_gb >= min_ram)
    if min_storage is not None:
        conditions.append(Variant.storage_gb >= min_storage)

    stmt = (
        select(Product.id)
        .join(Variant, Variant.product_id == Product.id)
        .where(and_(*conditions) if conditions else True)
        .distinct()
        .limit(max_candidates)
    )

    result = await db.execute(stmt)
    product_ids = [row[0] for row in result.fetchall()]
    logger.info("sql_filter_complete", candidate_count=len(product_ids))
    return product_ids


# ── Vector rerank ─────────────────────────────────────────────────────────────

async def vector_rerank_candidates(
    db: AsyncSession,
    filtered_product_ids: list[int],
    query_embedding: list[float],
    *,
    embedding_type: str = "specs",
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Step 2: Vector re-rank via JOIN on filtered IDs (HNSW, no full scan)."""
    settings = get_settings()
    max_candidates = max_candidates or settings.max_vector_candidates

    if not filtered_product_ids:
        return []

    if not query_embedding:
        return [{"product_id": pid, "distance": 1.0} for pid in filtered_product_ids[:max_candidates]]

    embedding_str = "[" + ",".join(str(f) for f in query_embedding) + "]"

    sql = text("""
        SELECT e.product_id,
               e.embedding_vector <=> :query_vec AS distance
        FROM embeddings e
        WHERE e.product_id = ANY(:product_ids)
          AND e.embedding_type = :emb_type
        ORDER BY e.embedding_vector <=> :query_vec
        LIMIT :max_count
    """)

    result = await db.execute(
        sql,
        {
            "query_vec": embedding_str,
            "product_ids": filtered_product_ids,
            "emb_type": embedding_type,
            "max_count": max_candidates,
        },
    )

    candidates = [
        {"product_id": row[0], "distance": float(row[1])}
        for row in result.fetchall()
    ]
    logger.info("vector_rerank_complete", candidate_count=len(candidates))
    return candidates


# ── Composite reranking ───────────────────────────────────────────────────────

def _compute_composite_score(
    candidate: dict[str, Any],
    *,
    budget_max: float | None,
    brands: list[str] | None,
) -> float:
    """Composite reranking score.

    score = 0.4 × semantic_similarity
          + 0.3 × price_relevance
          + 0.2 × brand_match
          + 0.1 × recency
    """
    # Semantic similarity (1 - distance, distance in [0,2] for cosine)
    distance = candidate.get("distance", 1.0)
    semantic = max(0.0, 1.0 - distance / 2.0)

    # Price relevance (1.0 if within budget, degraded if over)
    best_price = None
    for v in candidate.get("variants", []):
        p = v.get("price_inr")
        if p and (best_price is None or p < best_price):
            best_price = p

    price_rel = 1.0
    if budget_max and best_price:
        if best_price <= budget_max:
            price_rel = 1.0
        else:
            # Penalize over-budget products
            price_rel = max(0.0, 1.0 - (best_price - budget_max) / budget_max)
    elif best_price is None:
        price_rel = 0.5  # no price data

    # Brand match
    brand_match = 0.0
    if brands:
        product_brand = (candidate.get("brand") or "").lower()
        brand_match = 1.0 if any(b.lower() == product_brand for b in brands) else 0.0
    else:
        brand_match = 0.5  # no brand preference → neutral

    # Recency (use score_version as proxy: higher = more recently scored)
    score_version = candidate.get("scores", {}).get("score_version", 0) or 0
    recency = min(1.0, score_version / 5.0)  # cap at version 5

    score = 0.4 * semantic + 0.3 * price_rel + 0.2 * brand_match + 0.1 * recency
    return round(score, 4)


# ── Candidate reduction ───────────────────────────────────────────────────────

async def reduce_candidates(
    db: AsyncSession,
    ranked_candidates: list[dict[str, Any]],
    *,
    max_candidates: int | None = None,
    budget_max: float | None = None,
    brands: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Step 3: Fetch full product data + apply composite reranking."""
    settings = get_settings()
    max_candidates = max_candidates or settings.max_rank_candidates

    top_ids = [c["product_id"] for c in ranked_candidates[:max_candidates * 2]]  # fetch 2× for reranking
    if not top_ids:
        return []

    stmt = (
        select(Product, Variant, ComponentScore)
        .join(Variant, Variant.product_id == Product.id)
        .outerjoin(ComponentScore, ComponentScore.product_id == Product.id)
        .where(Product.id.in_(top_ids))
        .order_by(Variant.price_inr.asc())
    )

    result = await db.execute(stmt)
    rows = result.fetchall()

    candidates_map: dict[int, dict] = {}
    for product, variant, scores in rows:
        pid = product.id
        if pid not in candidates_map:
            distance = next(
                (c["distance"] for c in ranked_candidates if c["product_id"] == pid),
                1.0,
            )
            candidates_map[pid] = {
                "product_id": pid,
                "product_slug": product.product_slug,
                "brand": product.brand,
                "model_name": product.model_name,
                "canonical_title": product.canonical_title,
                "category": product.category.value,
                "vector_distance": distance,
                "specs": product.specs_jsonb,
                "variants": [],
                "scores": {
                    "performance_score": scores.performance_score if scores else 0.0,
                    "battery_score": scores.battery_score if scores else 0.0,
                    "camera_score": scores.camera_score if scores else 0.0,
                    "display_score": scores.display_score if scores else 0.0,
                    "build_score": scores.build_score if scores else 0.0,
                    "price_score": scores.price_score if scores else 0.0,
                    "popularity_score": scores.popularity_score if scores else 0.0,
                    "computed_at": scores.computed_at.isoformat() if scores and scores.computed_at else None,
                    "score_version": scores.score_version if scores else 0,
                },
            }
        candidates_map[pid]["variants"].append({
            "variant_id": variant.id,
            "ram_gb": variant.ram_gb,
            "storage_gb": variant.storage_gb,
            "color": variant.color,
            "price_inr": variant.price_inr,
            "availability": variant.availability_bool,
            "vendor": variant.vendor,
            "affiliate_url": variant.affiliate_url,
        })

    # Apply composite scoring
    for c in candidates_map.values():
        c["retrieval_score"] = _compute_composite_score(
            c, budget_max=budget_max, brands=brands
        )

    # Sort by composite score descending
    ordered = sorted(candidates_map.values(), key=lambda c: c["retrieval_score"], reverse=True)

    logger.info("candidate_reduction_complete", final_count=len(ordered[:max_candidates]))
    return ordered[:max_candidates]


# ── Cache check ───────────────────────────────────────────────────────────────

async def check_semantic_cache(
    redis_client: Any,
    embedding_hash: str,
    filter_hash: str,
) -> dict | None:
    """Check semantic cache for a matching result.

    Returns cached payload dict if found and fresh, None otherwise.
    """
    import orjson
    from app.services.cache import get_semantic_cache
    try:
        cached = await get_semantic_cache(redis_client, embedding_hash, filter_hash)
        return cached
    except Exception as e:
        logger.warning("cache_check_failed", error=str(e))
        return None


# ── Session bypass ────────────────────────────────────────────────────────────

async def check_session_bypass(
    session: Any,
    intent_type: str,
) -> list[dict] | None:
    """Return cached candidates from session for follow-up queries.

    Returns last_results if this is a follow-up intent and session has results.
    """
    if not session:
        return None
    is_followup = getattr(session, "turn_count", 0) > 1
    has_results = bool(getattr(session, "last_results", []))

    # Only bypass for certain follow-up intents, not for new searches
    bypass_intents = {"COMPARE_PRODUCTS", "FEATURE_QUERY"}
    if is_followup and has_results and intent_type in bypass_intents:
        logger.info(
            "retrieval_session_bypass",
            candidate_count=len(session.last_results),
            intent_type=intent_type,
        )
        return session.last_results

    return None
