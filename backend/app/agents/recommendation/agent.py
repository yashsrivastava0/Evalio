"""Recommendation Agent — LangGraph node.

Ranks candidates for the user's specific intent using AI reasoning.
Replaces the old deterministic 7-component scoring formula.

Uses gpt-4o for reasoning quality — recommendation ranking benefits from
understanding nuanced use-case requirements.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from app.agents.recommendation.schema import RecommendationResult
from app.agents.recommendation.tools import format_candidates_for_prompt
from app.core.logging import get_logger
from app.services.llm_router import get_model_for_agent

logger = get_logger(__name__)

_PROMPT_PATH = pathlib.Path(__file__).parent / "prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_MAX_RETRIES = 2


async def recommendation_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """LangGraph node: rank candidates and produce recommendation cards.

    Returns state update:
        recommendation_result: RecommendationResult (as dict)
    """
    query = state.get("query", "")
    intent = state.get("intent") or {}
    expanded = state.get("expanded_query") or {}
    candidates = state.get("candidates", [])
    review_consensus = state.get("review_consensus") or {}
    model = get_model_for_agent("recommendation")

    if isinstance(intent, dict):
        intent_type = intent.get("intent_type", "SEARCH_PRODUCT")
        focus = intent.get("focus_attributes", [])
        budget_max = intent.get("budget_max")
        category = intent.get("category", "")
    else:
        intent_type = getattr(intent, "intent_type", "SEARCH_PRODUCT")
        focus = getattr(intent, "focus_attributes", [])
        budget_max = getattr(intent, "budget_max", None)
        category = getattr(intent, "category", "")

    search_criteria = expanded.get("search_criteria", []) if isinstance(expanded, dict) else []

    if not candidates:
        logger.warning("recommendation_no_candidates")
        return {"recommendation_result": {
            "component": "recommendation_cards",
            "ui_hint": "render as cards",
            "data": {"ranked": [], "rationale": "No matching products found.", "user_intent_summary": query},
        }}

    candidates_text = format_candidates_for_prompt(candidates)

    # Include review consensus if available
    review_text = ""
    if review_consensus and review_consensus.get("data", {}).get("summaries"):
        review_text = "\n\nReal-world user feedback:\n"
        for slug, summary in review_consensus["data"]["summaries"].items():
            praise = ", ".join(summary.get("key_praise", []))
            complaints = ", ".join(summary.get("key_complaints", []))
            review_text += f"- {slug}: praise=[{praise}] complaints=[{complaints}]\n"

    user_message = (
        f"User query: \"{query}\"\n"
        f"Intent: {intent_type}\n"
        f"Category: {category or 'any'}\n"
        f"Budget max: {budget_max or 'no limit'} INR\n"
        f"Focus attributes: {focus}\n"
        f"Search criteria: {search_criteria}\n\n"
        f"Candidate products:\n{candidates_text}"
        f"{review_text}\n\n"
        f"Rank these products from best to worst for this user's specific need. "
        f"Return ONLY valid JSON matching the output schema."
    )

    result: RecommendationResult | None = None
    last_error = ""

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            result = RecommendationResult(**parsed)
            logger.info(
                "recommendation_complete",
                ranked_count=len(result.data.ranked),
            )
            break

        except Exception as e:
            last_error = str(e)
            logger.warning("recommendation_error", attempt=attempt, error=last_error)
            if attempt == _MAX_RETRIES:
                logger.error("recommendation_fallback", error=last_error)
                # Fallback: return candidates as-is with placeholder scores
                from app.agents.recommendation.schema import RecommendationData, RecommendedProduct, MatchScoreBreakdown
                from app.tools.price_tools import get_best_price
                ranked = [
                    RecommendedProduct(
                        product_id=c.get("product_id", 0),
                        product_slug=c.get("product_slug", ""),
                        canonical_title=c.get("canonical_title", "Unknown"),
                        match_score=round(1.0 - c.get("vector_distance", 0.5), 2),
                        match_score_breakdown=MatchScoreBreakdown(),
                        reason="Retrieved based on semantic similarity to your query.",
                        pros=[],
                        cons=[],
                        price=get_best_price(c.get("variants", [])),
                    )
                    for c in candidates[:5]
                ]
                result = RecommendationResult(
                    data=RecommendationData(
                        ranked=ranked,
                        rationale="Fallback: ranked by semantic similarity.",
                        user_intent_summary=query,
                    )
                )

    return {"recommendation_result": result.model_dump() if result else {}}
