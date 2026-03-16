"""Review Summary UI Contract.

Maps ReviewConsensus agent output → stable UI dict.
"""

from __future__ import annotations

from typing import Any


def to_ui(review_consensus: dict[str, Any]) -> dict[str, Any]:
    """Map ReviewConsensus → review_summary UI component."""
    data = review_consensus.get("data", {})
    summaries = data.get("summaries", {})

    return {
        "component": "review_summary",
        "version": "v1",
        "products": [
            {
                "slug": slug,
                "sentiment_score": s.get("sentiment_score"),
                "sentiment_label": s.get("sentiment_label", "unknown"),
                "source_count": s.get("source_count", 0),
                "key_praise": s.get("key_praise", []),
                "key_complaints": s.get("key_complaints", []),
            }
            for slug, s in summaries.items()
        ],
    }
