"""Query Expansion Agent — LangGraph node.

Transforms vague user queries into structured search criteria:
- Concrete technical terms (search_criteria)
- SQL filter parameters (budget, category, brands)
- Rewritten semantic search query (vector_query_text)

Also computes the query embedding for vector retrieval.
Uses gpt-4o-mini (fast — expansion doesn't need heavy reasoning).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from app.agents.query_expansion.schema import ExpandedQuery
from app.agents.query_expansion.tools import build_sql_filters_from_intent, expand_and_embed
from app.core.logging import get_logger
from app.services.llm_router import get_model_for_agent

logger = get_logger(__name__)

_PROMPT_PATH = pathlib.Path(__file__).parent / "prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_MAX_RETRIES = 2


async def query_expansion_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
    embedding_provider: Any,
) -> dict[str, Any]:
    """LangGraph node: expand query into structured search criteria.

    Returns state update:
        expanded_query: ExpandedQuery (as dict)
        query_embedding: list[float]
    """
    query = state.get("query", "")
    intent = state.get("intent") or {}
    model = get_model_for_agent("query_expansion")

    # Get intent type for context
    if isinstance(intent, dict):
        intent_type = intent.get("intent_type", "SEARCH_PRODUCT")
        focus = intent.get("focus_attributes", [])
        entities = intent.get("entities", [])
        budget_max = intent.get("budget_max")
        budget_min = intent.get("budget_min")
        category = intent.get("category")
        brands = intent.get("brands", [])
    else:
        intent_type = getattr(intent, "intent_type", "SEARCH_PRODUCT")
        focus = getattr(intent, "focus_attributes", [])
        entities = getattr(intent, "entities", [])
        budget_max = getattr(intent, "budget_max", None)
        budget_min = getattr(intent, "budget_min", None)
        category = getattr(intent, "category", None)
        brands = getattr(intent, "brands", [])

    user_message = (
        f"User query: \"{query}\"\n\n"
        f"Intent classified as: {intent_type}\n"
        f"Extracted entities: {entities}\n"
        f"Focus attributes: {focus}\n"
        f"Budget: min={budget_min} max={budget_max} INR\n"
        f"Category: {category}\n"
        f"Brands: {brands}\n\n"
        f"Expand this query into concrete search criteria and SQL filters. "
        f"Return ONLY valid JSON matching the output schema."
    )

    expanded: ExpandedQuery | None = None
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
                temperature=0.0,
                max_tokens=800,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)

            # Fill intent-derived sql_filters if LLM left them empty
            if not parsed.get("sql_filters") or parsed["sql_filters"] == {}:
                parsed["sql_filters"] = build_sql_filters_from_intent(intent if isinstance(intent, dict) else {})

            # Ensure sql_filters has intent budget/category if LLM missed them
            sf = parsed["sql_filters"]
            if sf.get("budget_max") is None and budget_max:
                sf["budget_max"] = budget_max
            if sf.get("budget_min") is None and budget_min:
                sf["budget_min"] = budget_min
            if sf.get("category") is None and category:
                sf["category"] = category
            if not sf.get("brands") and brands:
                sf["brands"] = brands

            parsed["intent_type"] = intent_type
            parsed.setdefault("original", query)

            expanded = ExpandedQuery(**parsed)
            logger.info(
                "query_expanded",
                criteria_count=len(expanded.search_criteria),
                has_sql_filters=bool(expanded.sql_filters),
            )
            break

        except Exception as e:
            last_error = str(e)
            logger.warning("query_expansion_error", attempt=attempt, error=last_error)
            if attempt == _MAX_RETRIES:
                logger.error("query_expansion_fallback", error=last_error)
                # Fallback: use raw query as expansion
                expanded = ExpandedQuery(
                    original=query,
                    search_criteria=[query],
                    vector_query_text=query,
                    intent_type=intent_type,
                    sql_filters=build_sql_filters_from_intent(intent if isinstance(intent, dict) else {}),
                )

    # Compute embedding for the expanded vector_query_text
    vector_text = expanded.vector_query_text if expanded else query
    query_embedding = await expand_and_embed(vector_text, embedding_provider)

    return {
        "expanded_query": expanded.model_dump() if expanded else {},
        "query_embedding": query_embedding,
    }
