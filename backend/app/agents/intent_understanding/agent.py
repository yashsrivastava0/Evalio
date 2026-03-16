"""Intent Understanding Agent — LangGraph node.

Classifies user intent and extracts structured entities, budget, brands,
and follow-up references using a long structured prompt + JSON mode.

Uses gpt-4o-mini (fast, cheap — intent classification doesn't need gpt-4o).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from app.agents.intent_understanding.schema import IntentResult
from app.agents.intent_understanding.tools import build_intent_prompt_context
from app.core.logging import get_logger
from app.services.llm_router import get_model_for_agent

logger = get_logger(__name__)

# Load prompt at module level (avoids repeated disk I/O)
_PROMPT_PATH = pathlib.Path(__file__).parent / "prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_MAX_RETRIES = 2


async def intent_understanding_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """LangGraph node: classify intent and extract structured entities.

    Returns state update:
        intent: IntentResult (as dict)
    """
    query = state.get("query", "")
    session = state.get("session")
    model = get_model_for_agent("intent")

    # Build prompt context
    ctx = build_intent_prompt_context(query, session)

    user_message = (
        f"User query: \"{query}\"\n\n"
        f"Session context:\n"
        f"- Previous conversation: {ctx['session_history']}\n"
        f"- Products seen this session: {ctx['session_products']}\n"
        f"- Last query: {ctx['last_query']}\n"
        f"- Turn count: {ctx['turn_count']}\n\n"
        f"Classify this query and extract structured information. "
        f"Return ONLY valid JSON matching the output schema."
    )

    intent_result = None
    last_error: str = ""

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
                max_tokens=512,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)

            # Validate against schema
            intent_result = IntentResult(**parsed)
            logger.info(
                "intent_classified",
                intent_type=intent_result.intent_type,
                entities=intent_result.entities,
                is_followup=intent_result.is_followup,
            )
            break

        except Exception as e:
            last_error = str(e)
            logger.warning("intent_agent_error", attempt=attempt, error=last_error)
            if attempt == _MAX_RETRIES:
                # Fallback: safe default intent
                logger.error("intent_agent_fallback", error=last_error)
                intent_result = IntentResult(
                    intent_type="SEARCH_PRODUCT",
                    raw_query=query,
                )

    return {"intent": intent_result.model_dump() if intent_result else {"intent_type": "SEARCH_PRODUCT", "raw_query": query}}
