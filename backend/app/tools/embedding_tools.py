"""Embedding tool functions shared across agents."""

from __future__ import annotations

from typing import Any

from app.agents.utils import compute_embedding_hash, normalize_query, compute_query_hash
from app.core.logging import get_logger

logger = get_logger(__name__)


async def embed_query(text: str, provider: Any) -> list[float]:
    """Embed a query string using the provided embedding provider.

    Returns empty list on failure — agents must handle gracefully.
    """
    if provider is None:
        logger.warning("embed_query_no_provider")
        return []
    try:
        embedding = await provider.embed(text)
        logger.info("query_embedded", dim=len(embedding), hash=compute_embedding_hash(embedding)[:8])
        return embedding
    except Exception as e:
        logger.error("embed_query_error", error=str(e))
        return []


def hash_query(query: str) -> tuple[str, str]:
    """Normalize and hash a query string.

    Returns (normalized_query, query_hash).
    """
    normalized = normalize_query(query)
    return normalized, compute_query_hash(normalized)
