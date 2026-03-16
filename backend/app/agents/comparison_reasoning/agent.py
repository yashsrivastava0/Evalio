"""Comparison Reasoning Agent — LangGraph node.

Generates a deep structured comparison between 2-4 products.
Replaces the old ComparisonGuard (numeric diffs) + Explanation (template prose).

Uses gpt-4o — comparison requires multi-product reasoning.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from app.agents.comparison_reasoning.schema import ComparisonResult
from app.agents.comparison_reasoning.tools import format_products_for_comparison
from app.core.logging import get_logger
from app.services.llm_router import get_model_for_agent

logger = get_logger(__name__)

_PROMPT_PATH = pathlib.Path(__file__).parent / "prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_MAX_RETRIES = 2


async def comparison_reasoning_node(
    state: dict[str, Any],
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """LangGraph node: produce structured product comparison.

    Returns state update:
        comparison_result: ComparisonResult (as dict)
    """
    query = state.get("query", "")
    intent = state.get("intent") or {}
    candidates = state.get("candidates", [])
    review_consensus = state.get("review_consensus") or {}
    model = get_model_for_agent("comparison_reasoning")

    if isinstance(intent, dict):
        focus = intent.get("focus_attributes", [])
    else:
        focus = getattr(intent, "focus_attributes", [])

    if not candidates:
        logger.warning("comparison_no_candidates")
        return {"comparison_result": {}}

    products_text = format_products_for_comparison(candidates)

    # Include review consensus if available
    review_text = ""
    if review_consensus and review_consensus.get("data", {}).get("summaries"):
        review_text = "\n\nReal-world user feedback:\n"
        for slug, summary in review_consensus["data"]["summaries"].items():
            praise = ", ".join(summary.get("key_praise", []))
            complaints = ", ".join(summary.get("key_complaints", []))
            review_text += f"- {slug}: users praise=[{praise}], users complain=[{complaints}]\n"

    user_message = (
        f"User query: \"{query}\"\n"
        f"User focus: {focus}\n\n"
        f"Products to compare:\n{products_text}"
        f"{review_text}\n\n"
        f"Generate a deep technical comparison. "
        f"Return ONLY valid JSON matching the output schema."
    )

    result: ComparisonResult | None = None
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
                max_tokens=3000,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            result = ComparisonResult(**parsed)
            logger.info(
                "comparison_complete",
                product_count=len(result.data.products),
                diff_count=len(result.data.technical_diffs),
            )
            break

        except Exception as e:
            last_error = str(e)
            logger.warning("comparison_error", attempt=attempt, error=last_error)
            if attempt == _MAX_RETRIES:
                logger.error("comparison_fallback", error=last_error)

    return {"comparison_result": result.model_dump() if result else {}}
