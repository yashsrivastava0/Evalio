"""Comparison Table UI Contract.

Maps ComparisonResult agent output → stable UI dict that the frontend
can depend on across agent schema changes.
"""

from __future__ import annotations

from typing import Any


def to_ui(comparison_result: dict[str, Any]) -> dict[str, Any]:
    """Map ComparisonResult → comparison_table UI component."""
    data = comparison_result.get("data", {})
    return {
        "component": "comparison_table",
        "version": "v1",
        "products": [
            {
                "id": p.get("product_id"),
                "slug": p.get("product_slug"),
                "title": p.get("canonical_title"),
                "price": p.get("price_range"),
                "pros": p.get("pros", []),
                "cons": p.get("cons", []),
            }
            for p in data.get("products", [])
        ],
        "technical_diffs": [
            {
                "attribute": d.get("attribute"),
                "values": d.get("values", {}),
                "winner": d.get("winner"),
                "explanation": d.get("explanation"),
            }
            for d in data.get("technical_diffs", [])
        ],
        "per_category_winners": data.get("per_category_winners", {}),
        "overall_recommendation": data.get("overall_recommendation", ""),
        "tradeoffs": data.get("tradeoffs", []),
        "use_case_suitability": data.get("use_case_suitability", {}),
    }
