"""Review Consensus Agent — Tool functions."""

from __future__ import annotations

from typing import Any

from app.tools.db_tools import fetch_review_summaries as _fetch_reviews


async def fetch_review_summaries(
    db: Any,
    product_ids: list[int],
) -> list[dict[str, Any]]:
    """Fetch ReviewSummary rows for candidate products."""
    return await _fetch_reviews(db, product_ids)


def parse_sentiment_into_label(score: float | None) -> str:
    """Convert sentiment score to human-readable label."""
    if score is None:
        return "unknown"
    if score >= 0.7:
        return "positive"
    if score >= 0.4:
        return "mixed"
    return "negative"


def build_review_prompt_text(reviews: list[dict[str, Any]]) -> str:
    """Format review data for the LLM prompt."""
    if not reviews:
        return "No review data available."

    lines: list[str] = []
    for r in reviews:
        lines.append(
            f"Product ID {r['product_id']} (slug to be resolved from candidates):\n"
            f"  Summary: {r.get('summary_text', 'N/A')[:500]}\n"
            f"  Sentiment: {r.get('sentiment_score', 'N/A')}\n"
            f"  Sources: {r.get('source_count', 0)}\n"
            f"  Last synced: {r.get('last_synced', 'N/A')}"
        )
    return "\n\n".join(lines)
