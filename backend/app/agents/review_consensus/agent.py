"""Review Consensus Agent — LangGraph node.

Reads ReviewSummary rows from DB and uses LLM to extract structured
key_praise and key_complaints per product.

Graceful no-op if no review data exists in DB.
Uses gpt-4o — extracting meaningful nuance from review text benefits from stronger reasoning.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.review_consensus.schema import ReviewConsensus, ReviewConsensusData, ProductReviewSummary
from app.agents.review_consensus.tools import fetch_review_summaries, parse_sentiment_into_label, build_review_prompt_text
from app.core.logging import get_logger
from app.services.llm_router import get_model_for_agent

logger = get_logger(__name__)

_PROMPT_PATH = pathlib.Path(__file__).parent / "prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_MAX_RETRIES = 2


async def review_consensus_node(
    state: dict[str, Any],
    *,
    db_session: AsyncSession,
    llm_client: Any,
) -> dict[str, Any]:
    """LangGraph node: fetch and summarize review data for candidates.

    Returns state update:
        review_consensus: ReviewConsensus (as dict)
    """
    candidates = state.get("candidates", [])
    model = get_model_for_agent("review_consensus")

    # Build product_id → product_slug mapping
    id_to_slug: dict[int, str] = {
        c.get("product_id", 0): c.get("product_slug", "")
        for c in candidates
        if c.get("product_id")
    }
    product_ids = list(id_to_slug.keys())

    # Fetch review data from DB
    reviews = await fetch_review_summaries(db_session, product_ids)

    # Graceful no-op: no review data in DB
    if not reviews:
        logger.info("review_consensus_no_data", product_count=len(product_ids))
        empty = ReviewConsensus(data=ReviewConsensusData())
        return {"review_consensus": empty.model_dump()}

    # Add slug to review rows
    for r in reviews:
        r["product_slug"] = id_to_slug.get(r.get("product_id", 0), "")

    review_text = build_review_prompt_text(reviews)

    # Build slug mapping hint for LLM
    slug_map = "\n".join(
        f"  product_id={r['product_id']} → slug='{r['product_slug']}'"
        for r in reviews
    )

    user_message = (
        f"Product ID to slug mapping:\n{slug_map}\n\n"
        f"Review data:\n{review_text}\n\n"
        f"Extract key_praise (2-3 items) and key_complaints (1-2 items) for each product. "
        f"Be specific and concrete. Use the product_slug as the key in the summaries dict. "
        f"Return ONLY valid JSON matching the output schema."
    )

    result: ReviewConsensus | None = None
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
                temperature=0.1,
                max_tokens=1500,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            result = ReviewConsensus(**parsed)
            logger.info("review_consensus_complete", product_count=len(result.data.summaries))
            break

        except Exception as e:
            last_error = str(e)
            logger.warning("review_consensus_error", attempt=attempt, error=last_error)
            if attempt == _MAX_RETRIES:
                # Fallback: build basic summary from DB data without LLM
                logger.warning("review_consensus_fallback", error=last_error)
                summaries: dict[str, ProductReviewSummary] = {}
                for r in reviews:
                    slug = r.get("product_slug", "")
                    if slug:
                        summaries[slug] = ProductReviewSummary(
                            product_slug=slug,
                            sentiment_score=r.get("sentiment_score"),
                            sentiment_label=parse_sentiment_into_label(r.get("sentiment_score")),
                            summary_text=r.get("summary_text", ""),
                            source_count=r.get("source_count", 0),
                            last_synced=r.get("last_synced"),
                            key_praise=[],
                            key_complaints=[],
                        )
                result = ReviewConsensus(data=ReviewConsensusData(summaries=summaries))

    return {"review_consensus": result.model_dump() if result else ReviewConsensus(data=ReviewConsensusData()).model_dump()}
