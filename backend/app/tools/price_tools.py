"""Price tool functions shared across agents."""

from __future__ import annotations

from typing import Any


def get_best_price(variants: list[dict[str, Any]]) -> float | None:
    """Return the lowest available price across all variants.

    Returns None if no available variants with prices exist.
    """
    available_prices = [
        v["price_inr"]
        for v in variants
        if v.get("availability") and v.get("price_inr") is not None
    ]
    return min(available_prices) if available_prices else None


def get_price_range(variants: list[dict[str, Any]]) -> str:
    """Return a human-readable price range string."""
    available = [
        v["price_inr"]
        for v in variants
        if v.get("availability") and v.get("price_inr") is not None
    ]
    if not available:
        return "Price unavailable"
    low, high = min(available), max(available)
    if low == high:
        return f"₹{low:,.0f}"
    return f"₹{low:,.0f} – ₹{high:,.0f}"


def extract_price_summary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a structured price summary dict for LLM prompts."""
    available = [v for v in variants if v.get("availability") and v.get("price_inr")]
    all_prices = [v["price_inr"] for v in available]
    vendors = list({v.get("vendor", "Unknown") for v in available})

    return {
        "best_price": min(all_prices) if all_prices else None,
        "max_price": max(all_prices) if all_prices else None,
        "vendor_count": len(vendors),
        "vendors": vendors,
        "variant_count": len(variants),
        "available_variants": len(available),
        "display": get_price_range(variants),
    }
