"""Recommendation Agent — Tool functions."""

from __future__ import annotations

from typing import Any

from app.tools.price_tools import get_best_price, extract_price_summary
from app.tools.spec_tools import get_normalized_specs, format_specs_for_prompt


def format_candidate_for_ranking(candidate: dict[str, Any]) -> dict[str, Any]:
    """Trim candidate to fields needed for recommendation prompt.

    Excludes large raw specs — uses normalized canonical values instead.
    """
    specs_raw = candidate.get("specs") or {}
    canonical = get_normalized_specs(specs_raw)
    specs_text = format_specs_for_prompt(specs_raw, canonical)

    variants = candidate.get("variants", [])
    best_price = get_best_price(variants)
    price_summary = extract_price_summary(variants)

    return {
        "product_id": candidate.get("product_id"),
        "product_slug": candidate.get("product_slug"),
        "canonical_title": candidate.get("canonical_title"),
        "brand": candidate.get("brand"),
        "category": candidate.get("category"),
        "specs_summary": specs_text,
        "normalized_specs": canonical,
        "best_price": best_price,
        "price_display": price_summary.get("display", "N/A"),
        "scores": candidate.get("scores", {}),
        "retrieval_score": candidate.get("retrieval_score", 0.0),
    }


def format_candidates_for_prompt(
    candidates: list[dict[str, Any]],
    max_candidates: int = 8,
) -> str:
    """Format top candidates as a numbered list for the LLM prompt."""
    lines: list[str] = []
    for i, c in enumerate(candidates[:max_candidates], 1):
        formatted = format_candidate_for_ranking(c)
        lines.append(
            f"Product {i}: {formatted['canonical_title']} "
            f"(slug: {formatted['product_slug']}, id: {formatted['product_id']})\n"
            f"  Price: {formatted['price_display']}\n"
            f"  Specs: {formatted['specs_summary']}"
        )
    return "\n\n".join(lines)
