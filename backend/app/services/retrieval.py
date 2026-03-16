"""Retrieval Service — SQL-first query planner + pgvector re-rank.

Fast Path (corrected order):
    1. SQL filter → reduce ~5000 → ~300 filtered IDs
    2. Vector re-rank via JOIN on filtered IDs (HNSW, no full scan)
    3. Final candidate reduction → top max_rank_candidates for ranking

Vector re-rank uses explicit JOIN:
    SELECT ... FROM embeddings e
    JOIN filtered_ids fp ON e.product_id = fp.id
    WHERE e.embedding_type = :type   -- Phase 6: required, prevents cross-type scan
    ORDER BY e.embedding_vector <=> :query_vector
    LIMIT max_vector_candidates

Phase 6: select_embedding_type(intent) lives HERE (RetrievalNode owns candidate selection).
    QueryEmbeddingNode stays pure: query → vector → hash only.

Fetch Path is handled by DecisionMatrix → async Celery task (never blocks).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.tables import Product, Variant, ComponentScore, Embedding, EmbeddingType

logger = get_logger(__name__)


# ── Phase 6: Deterministic Embedding Type Selection ───────────────────────────
# Lives in retrieval.py — RetrievalNode owns candidate selection routing.
# Never uses LLM. Pure rule-based function. Deterministic.

_REVIEW_PRIORITY_KEYWORDS: frozenset[str] = frozenset({
    "camera", "battery life", "battery", "heating", "heat",
    "review", "reviews", "durability", "build quality",
})


def select_embedding_type(intent: Any) -> str:
    """Deterministically select embedding type from intent.

    Returns "review" when:
        - Intent priority keywords match review-signal terms (camera, battery, etc.)
        - intent_type == "product_qa"
    Returns "specs" in all other cases (default).

    NO LLM, NO cosine, NO external calls. Pure deterministic rules.
    Owned by RetrievalNode — QueryEmbeddingNode is NOT modified.
    """
    if intent is None:
        return "specs"

    priorities = getattr(intent, "priority", []) or []
    intent_type = getattr(intent, "intent_type", "search") or "search"

    # Normalize to lowercase for comparison
    priority_set = {p.lower() for p in priorities if isinstance(p, str)}

    if priority_set & _REVIEW_PRIORITY_KEYWORDS:
        return "review"
    if intent_type == "product_qa":
        return "review"
    return "specs"



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
    """Step 1: SQL parameterized filter → reduce to ~300 candidate product IDs.

    Uses composite index on (availability_bool, price_inr).
    Returns product IDs only — no data fetched yet.
    """
    settings = get_settings()
    max_candidates = max_candidates or settings.max_sql_candidates

    # Build parameterized query
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


async def vector_rerank_candidates(
    db: AsyncSession,
    filtered_product_ids: list[int],
    query_embedding: list[float],
    *,
    embedding_type: str = "specs",
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Step 2: Vector re-rank via JOIN on filtered IDs.

    CANDIDATE POOL ORDERING GUARANTEE (Fix-C confirmed):
        SQL filter runs FIRST (sql_filter_candidates) → shrinks ~5000 to ~300 IDs.
        This function receives already-filtered IDs as input.
        WHERE e.product_id = ANY(:product_ids) scopes the vector scan ONLY to
        SQL-approved candidates — NOT the full embeddings table.

        embedding_type filter applies WITHIN that restricted set.
        It never shrinks the pre-SQL candidate pool.

        Consequence: if only 10% of products have review embeddings,
        those products get higher cosine distance and sort lower —
        they are not excluded before vector re-rank.
        SQL recall is preserved. Vector re-rank only re-orders, never pre-filters.

    Uses HNSW index — no full table scan.
    Returns product IDs with cosine distances, ordered by similarity.
    """
    settings = get_settings()
    max_candidates = max_candidates or settings.max_vector_candidates

    if not filtered_product_ids:
        return []

    if not query_embedding:
        # No embedding available — return filtered IDs without vector ranking
        return [{"product_id": pid, "distance": 1.0} for pid in filtered_product_ids[:max_candidates]]

    # Build the vector re-rank query with explicit JOIN
    # Convert embedding to pgvector format
    embedding_str = "[" + ",".join(str(f) for f in query_embedding) + "]"

    # Use parameterized query with JOIN on filtered IDs
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


async def reduce_candidates(
    db: AsyncSession,
    ranked_candidates: list[dict[str, Any]],
    *,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Step 3: Final candidate reduction → top N for deterministic ranking.

    Fetches full product + variant + component_score data for final candidates.
    """
    settings = get_settings()
    max_candidates = max_candidates or settings.max_rank_candidates

    top_ids = [c["product_id"] for c in ranked_candidates[:max_candidates]]

    if not top_ids:
        return []

    # Fetch full product data
    stmt = (
        select(Product, Variant, ComponentScore)
        .join(Variant, Variant.product_id == Product.id)
        .outerjoin(ComponentScore, ComponentScore.product_id == Product.id)
        .where(Product.id.in_(top_ids))
        .order_by(Variant.price_inr.asc())
    )

    result = await db.execute(stmt)
    rows = result.fetchall()

    # Build candidate dicts with all data needed for ranking
    candidates_map: dict[int, dict] = {}
    for product, variant, scores in rows:
        pid = product.id
        if pid not in candidates_map:
            # Find the distance from vector ranking
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

    # Preserve vector ranking order
    ordered = [candidates_map[c["product_id"]] for c in ranked_candidates[:max_candidates] if c["product_id"] in candidates_map]

    logger.info("candidate_reduction_complete", final_count=len(ordered))
    return ordered


async def resolve_comparison_entities(
    db: AsyncSession,
    slugs: list[str],
) -> list[int]:
    """Resolve comparison entity slugs to product IDs.

    Tries in order:
        1. Exact product_slug match
        2. canonical_title ILIKE match (title contains the slug tokens)
    Returns list of matched product_ids (may be partial or empty).
    """
    if not slugs:
        return []

    matched_ids: list[int] = []
    seen: set[int] = set()

    for slug in slugs:
        # 1. Exact slug match
        result = await db.execute(
            select(Product.id).where(Product.product_slug == slug)
        )
        row = result.first()
        if row and row[0] not in seen:
            matched_ids.append(row[0])
            seen.add(row[0])
            continue

        # 2. Canonical title ILIKE: treat slug tokens as search terms
        tokens = slug.replace("-", " ").strip()
        if tokens:
            result = await db.execute(
                select(Product.id).where(
                    Product.canonical_title.ilike(f"%{tokens}%")
                ).limit(1)
            )
            row = result.first()
            if row and row[0] not in seen:
                matched_ids.append(row[0])
                seen.add(row[0])

    if matched_ids:
        logger.info("comparison_entity_resolved", slugs=slugs, matched_ids=matched_ids)
    else:
        logger.info("comparison_entity_not_found", slugs=slugs)

    return matched_ids


# ── LangGraph Node Function ──

async def retrieval_node(state: dict, *, db_session: AsyncSession) -> dict:
    """LangGraph node: Full retrieval pipeline.

    SQL filter → Vector re-rank → Candidate reduction.
    Pure node — returns state update only.

    Phase 6: select_embedding_type(intent) called here, NOT in QueryEmbeddingNode.
    Embedding type is set in state for downstream visibility and metrics.

    Session bypass: if DecisionMatrix injected candidates via session
    follow-up, skip the entire SQL+vector pipeline.
    """
    # Session bypass — follow-up queries reuse previous results
    if state.get("use_session_comparison") and state.get("candidates"):
        logger.info(
            "retrieval_session_bypass",
            candidate_count=len(state["candidates"]),
        )
        return {"candidates": state["candidates"]}

    intent = state.get("intent")
    query_embedding = state.get("query_embedding", [])
    settings = get_settings()

    # Phase 6: Select embedding type based on intent (deterministic, no LLM)
    embedding_type = select_embedding_type(intent)
    logger.info("retrieval_embedding_type_selected", embedding_type=embedding_type)

    # Entity resolution: for comparison queries with named models, resolve by slug/title first.
    # If found, skip the SQL filter (use resolved IDs directly for vector re-rank).
    comparison_slugs = getattr(intent, "comparison_slugs", []) or []
    if comparison_slugs and getattr(intent, "is_comparison", False):
        resolved_ids = await resolve_comparison_entities(db_session, comparison_slugs)
        if resolved_ids:
            logger.info("retrieval_entity_resolved", count=len(resolved_ids))
            vector_ranked = await vector_rerank_candidates(
                db_session, resolved_ids, query_embedding, embedding_type=embedding_type,
            )
            candidates = await reduce_candidates(db_session, vector_ranked)
            return {"candidates": candidates, "embedding_type": embedding_type}

    # Step 1: SQL filter
    product_ids = await sql_filter_candidates(
        db_session,
        category=getattr(intent, "category", None),
        budget_min=getattr(intent, "budget_min", None),
        budget_max=getattr(intent, "budget_max", None),
        brands=getattr(intent, "brands", None),
        min_ram=getattr(intent, "min_ram", None),
        min_storage=getattr(intent, "min_storage", None),
    )

    # Comparison broadening: if brand filter returned 0 results for a compare
    # intent, retry without the brand constraint so we can show alternatives.
    intent_brands = getattr(intent, "brands", []) or []
    catalog_miss_slugs: list[str] = []
    if not product_ids and getattr(intent, "is_comparison", False) and intent_brands:
        # Signal that the requested products were not found — caller surfaces this to user.
        comparison_slugs_missed = getattr(intent, "comparison_slugs", []) or []
        if comparison_slugs_missed:
            catalog_miss_slugs = comparison_slugs_missed
        logger.info("retrieval_compare_broadening", dropped_brands=intent_brands)
        product_ids = await sql_filter_candidates(
            db_session,
            category=getattr(intent, "category", None),
            budget_min=getattr(intent, "budget_min", None),
            budget_max=getattr(intent, "budget_max", None),
            brands=None,
            min_ram=getattr(intent, "min_ram", None),
            min_storage=getattr(intent, "min_storage", None),
        )

    # Step 2: Vector re-rank via JOIN on filtered IDs
    # Passes embedding_type — SQL enforces WHERE embedding_type = :type
    vector_ranked = await vector_rerank_candidates(
        db_session,
        product_ids,
        query_embedding,
        embedding_type=embedding_type,
    )

    # If review embedding yields no results, fall back to specs silently
    if not vector_ranked and embedding_type != "specs":
        logger.info(
            "retrieval_embedding_type_fallback",
            requested=embedding_type,
            fallback="specs",
        )
        embedding_type = "specs"
        vector_ranked = await vector_rerank_candidates(
            db_session,
            product_ids,
            query_embedding,
            embedding_type="specs",
        )

    # Step 3: Candidate reduction
    candidates = await reduce_candidates(db_session, vector_ranked)

    # Phase 6: Return embedding_type in state for downstream visibility
    result: dict = {"candidates": candidates, "embedding_type": embedding_type}
    if catalog_miss_slugs:
        result["catalog_miss_slugs"] = catalog_miss_slugs
    return result
