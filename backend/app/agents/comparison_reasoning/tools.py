"""Comparison Reasoning Agent — Tool functions."""

from __future__ import annotations

from typing import Any

from app.tools.price_tools import get_price_range, extract_price_summary
from app.tools.spec_tools import get_normalized_specs, format_specs_for_prompt


def format_products_for_comparison(candidates: list[dict[str, Any]]) -> str:
    """Format candidate products for comparison prompt.

    Returns a structured multi-product text representation.
    """
    sections: list[str] = []
    for c in candidates:
        specs_raw = c.get("specs") or {}
        canonical = get_normalized_specs(specs_raw)
        specs_text = format_specs_for_prompt(specs_raw, canonical)
        price_range = get_price_range(c.get("variants", []))
        scores = c.get("scores", {})

        sections.append(
            f"=== {c.get('canonical_title', 'Unknown')} ===\n"
            f"Slug: {c.get('product_slug', '')}\n"
            f"ID: {c.get('product_id', 0)}\n"
            f"Price: {price_range}\n"
            f"Specs:\n{specs_text}\n"
            f"Component scores: performance={scores.get('performance_score', 'N/A')} "
            f"battery={scores.get('battery_score', 'N/A')} "
            f"camera={scores.get('camera_score', 'N/A')} "
            f"display={scores.get('display_score', 'N/A')}"
        )
    return "\n\n".join(sections)
