"""Structured Output Generator.

Pure Python node — no LLM. Assembles agent outputs from GraphState into
the final StructuredOutput payload for the API layer.

The order of components matters for UI rendering:
1. Comparison table (if present)
2. Recommendation cards (if present)
3. Product detail (if present)
4. Review summary (if present and has data)
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


async def structured_output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble all agent outputs into a single StructuredOutput dict."""
    intent = state.get("intent") or {}
    intent_type = intent.get("intent_type", "") if isinstance(intent, dict) else ""

    components: list[dict[str, Any]] = []

    # 1. Comparison table
    comparison = state.get("comparison_result")
    if comparison and isinstance(comparison, dict) and comparison.get("data"):
        components.append(comparison)

    # 2. Recommendation cards
    recommendation = state.get("recommendation_result")
    if recommendation and isinstance(recommendation, dict) and recommendation.get("data"):
        ranked = recommendation.get("data", {}).get("ranked", [])
        if ranked:
            components.append(recommendation)

    # 3. Product detail
    knowledge = state.get("product_knowledge_result")
    if knowledge and isinstance(knowledge, dict) and knowledge.get("data"):
        components.append(knowledge)

    # 4. Review summary (only append if has actual summaries)
    review = state.get("review_consensus")
    if review and isinstance(review, dict):
        summaries = review.get("data", {}).get("summaries", {})
        if summaries:
            components.append(review)

    # Build metadata: confidence, sources, retrieval method
    candidates = state.get("candidates") or []
    retrieval_route = state.get("retrieval_route") or "unknown"
    rec = state.get("recommendation_result") or {}
    ranked = (rec.get("data") or {}).get("ranked") or []
    if ranked:
        avg = float(sum(p.get("match_score", 0.0) for p in ranked)) / len(ranked)
        confidence = int(avg * 100) / 100
    else:
        confidence = 0.0
    sources = list({c.get("source", "") for c in candidates if c.get("source")})
    metadata: dict[str, Any] = {
        "confidence_score": confidence,
        "sources": sources,
        "retrieval_method": retrieval_route,
    }

    output: dict[str, Any] = {
        "version": "v1",
        "session_id": state.get("session_id", ""),
        "trace_id": state.get("trace_id", ""),
        "intent_type": intent_type,
        "components": components,
        "metadata": metadata,
    }

    logger.info(
        "structured_output_assembled",
        intent_type=intent_type,
        component_count=len(components),
        component_types=[c.get("component", "?") for c in components],
    )

    return {"structured_output": output}
