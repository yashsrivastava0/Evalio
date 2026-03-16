"""Query Expansion Agent — Tool functions."""

from __future__ import annotations

from typing import Any

from app.tools.embedding_tools import embed_query


async def expand_and_embed(
    vector_query_text: str,
    embedding_provider: Any,
) -> list[float]:
    """Embed the expanded vector_query_text for retrieval."""
    return await embed_query(vector_query_text, embedding_provider)


def build_sql_filters_from_intent(intent: dict[str, Any]) -> dict[str, Any]:
    """Convert IntentResult dict into sql_filters structure.

    Used as a fallback if the LLM does not produce sql_filters.
    """
    return {
        "category": intent.get("category"),
        "budget_min": intent.get("budget_min"),
        "budget_max": intent.get("budget_max"),
        "brands": intent.get("brands") or [],
        "min_ram": None,
        "min_storage": None,
    }
