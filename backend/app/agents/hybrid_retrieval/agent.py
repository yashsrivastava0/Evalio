"""Hybrid Retrieval Agent — LangGraph node.

Pipeline:
    1. Check session bypass (follow-up queries reuse previous candidates)
    2. Check semantic cache
    3. Resolve entities (for COMPARE_PRODUCTS with named products)
    4. SQL filter candidates by budget/brand/category
    5. Select embedding type (specs vs review)
    6. Vector rerank via pgvector HNSW
    7. Composite score reranking (semantic + price + brand + recency)
    8. Deduplication
    9. Return candidates + retrieval_route for graph branching

No LLM calls — pure DB + cache logic.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.hybrid_retrieval.tools import (
    check_semantic_cache,
    check_session_bypass,
    reduce_candidates,
    select_embedding_type,
    sql_filter_candidates,
    vector_rerank_candidates,
)
from app.agents.utils import compute_embedding_hash, compute_filter_hash
from app.core.logging import get_logger
from app.services.entity_resolver import resolve_product_entities, resolve_to_product_ids
from app.services.product_deduplicator import deduplicate_candidates

logger = get_logger(__name__)


async def hybrid_retrieval_node(
    state: dict[str, Any],
    *,
    db_session: AsyncSession,
    redis_client: Any,
) -> dict[str, Any]:
    """LangGraph node: Full hybrid retrieval pipeline.

    Returns state update keys:
        candidates, retrieval_route, cache_payload, fetch_needed,
        query_embedding (if not already set)
    """
    intent = state.get("intent") or {}
    expanded_query = state.get("expanded_query") or {}
    session = state.get("session")
    query_embedding = state.get("query_embedding", [])

    # Normalize intent fields (supports both dict and dataclass)
    if isinstance(intent, dict):
        intent_type = intent.get("intent_type", "SEARCH_PRODUCT")
        comparison_targets = intent.get("comparison_targets", []) or []
        is_followup = intent.get("is_followup", False)
        entities = intent.get("entities", [])
    else:
        intent_type = getattr(intent, "intent_type", "SEARCH_PRODUCT")
        comparison_targets = getattr(intent, "comparison_targets", []) or []
        is_followup = getattr(intent, "is_followup", False)
        entities = getattr(intent, "entities", [])

    if isinstance(expanded_query, dict):
        sql_filters = expanded_query.get("sql_filters", {}) or {}
    else:
        sql_filters = getattr(expanded_query, "sql_filters", {}) or {}

    # ── 1. Session bypass ─────────────────────────────────────────────────────
    if is_followup:
        bypassed = await check_session_bypass(session, intent_type)
        if bypassed:
            return {
                "candidates": bypassed,
                "retrieval_route": "session_bypass",
                "fetch_needed": False,
            }

    # ── 2. Semantic cache check ────────────────────────────────────────────────
    if query_embedding:
        emb_hash = compute_embedding_hash(query_embedding)
        filter_hash = compute_filter_hash(sql_filters)
        cached = await check_semantic_cache(redis_client, emb_hash, filter_hash)
        if cached:
            logger.info("retrieval_cache_hit")
            return {
                "candidates": [],
                "retrieval_route": "cache_hit",
                "cache_payload": cached,
                "fetch_needed": False,
            }

    # ── 3. Entity resolution (COMPARE intent) ────────────────────────────────
    target_ids: list[int] = []
    if intent_type == "COMPARE_PRODUCTS" and (comparison_targets or entities):
        targets = comparison_targets or entities
        resolved_slugs = await resolve_product_entities(targets, db_session)
        if resolved_slugs:
            target_ids = await resolve_to_product_ids(resolved_slugs, db_session)
            logger.info("entity_resolved", slugs=resolved_slugs, ids=target_ids)

    # ── 4. SQL filter ─────────────────────────────────────────────────────────
    if target_ids:
        # For COMPARE with entity resolution: skip SQL filter, use resolved IDs
        product_ids = target_ids
        retrieval_route = "entity_resolved"
    else:
        product_ids = await sql_filter_candidates(
            db_session,
            category=sql_filters.get("category"),
            budget_min=sql_filters.get("budget_min"),
            budget_max=sql_filters.get("budget_max"),
            brands=sql_filters.get("brands") or [],
            min_ram=sql_filters.get("min_ram"),
            min_storage=sql_filters.get("min_storage"),
        )
        retrieval_route = "sql_vector"

        # Broadening: no results with brand filter → retry without brand
        if not product_ids and sql_filters.get("brands"):
            logger.info("retrieval_broadening_brands")
            product_ids = await sql_filter_candidates(
                db_session,
                category=sql_filters.get("category"),
                budget_min=sql_filters.get("budget_min"),
                budget_max=sql_filters.get("budget_max"),
                brands=None,
                min_ram=sql_filters.get("min_ram"),
                min_storage=sql_filters.get("min_storage"),
            )

    # ── 5. Check DB miss → fetch trigger ─────────────────────────────────────
    if not product_ids:
        logger.info("retrieval_db_miss", intent_type=intent_type)
        return {
            "candidates": [],
            "retrieval_route": "fetch_trigger",
            "fetch_needed": True,
        }

    # ── 6. Select embedding type ──────────────────────────────────────────────
    focus_attributes: list[str] = []
    if isinstance(expanded_query, dict):
        focus_attributes = expanded_query.get("search_criteria", [])
    else:
        focus_attributes = getattr(expanded_query, "search_criteria", [])
    # Also include intent focus_attributes
    if isinstance(intent, dict):
        focus_attributes += intent.get("focus_attributes", [])
    else:
        focus_attributes += getattr(intent, "focus_attributes", [])

    embedding_type = select_embedding_type(focus_attributes, intent_type)

    # ── 7. Vector rerank ──────────────────────────────────────────────────────
    vector_ranked = await vector_rerank_candidates(
        db_session, product_ids, query_embedding, embedding_type=embedding_type,
    )

    # Fallback to specs if no review embeddings exist
    if not vector_ranked and embedding_type != "specs":
        logger.info("retrieval_embedding_fallback", from_type=embedding_type, to_type="specs")
        embedding_type = "specs"
        vector_ranked = await vector_rerank_candidates(
            db_session, product_ids, query_embedding, embedding_type="specs",
        )

    # ── 8. Candidate reduction + composite scoring ────────────────────────────
    candidates_raw = await reduce_candidates(
        db_session,
        vector_ranked,
        budget_max=sql_filters.get("budget_max"),
        brands=sql_filters.get("brands") or [],
    )

    # ── 9. Deduplication ──────────────────────────────────────────────────────
    candidates = await deduplicate_candidates(candidates_raw)

    return {
        "candidates": candidates,
        "retrieval_route": retrieval_route,
        "fetch_needed": False,
        "cache_payload": None,
    }
