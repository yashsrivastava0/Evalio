"""Recommendation Cards UI Contract.

Maps RecommendationResult agent output → stable UI dict.
"""

from __future__ import annotations

from typing import Any


def to_ui(recommendation_result: dict[str, Any]) -> dict[str, Any]:
    """Map RecommendationResult → recommendation_cards UI component."""
    data = recommendation_result.get("data", {})
    return {
        "component": "recommendation_cards",
        "version": "v1",
        "products": [
            {
                "id": p.get("product_id"),
                "slug": p.get("product_slug"),
                "title": p.get("canonical_title"),
                "match_score": p.get("match_score"),
                "match_score_breakdown": p.get("match_score_breakdown", {}),
                "reason": p.get("reason"),
                "pros": p.get("pros", []),
                "cons": p.get("cons", []),
                "price": p.get("price"),
            }
            for p in data.get("ranked", [])
        ],
        "rationale": data.get("rationale", ""),
        "user_intent_summary": data.get("user_intent_summary", ""),
    }
