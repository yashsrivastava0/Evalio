"""Product Deduplicator Service.

Multiple scrape sources (Amazon, Flipkart, Smartprix) can produce near-duplicate
product entries for the same physical product:
    - iphone-15
    - iphone-15-global
    - iphone-15-india

This service consolidates duplicates using brand + model_name + category matching,
keeping the canonical entry (lowest product_id as tiebreaker) and merging variants
to get the best available price.

Called by: Hybrid Retrieval Agent after reduce_candidates().
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def _canonical_key(candidate: dict[str, Any]) -> str:
    """Generate a deduplication key from brand + normalized model name + category."""
    brand = (candidate.get("brand") or "").lower().strip()
    model = (candidate.get("model_name") or candidate.get("canonical_title") or "").lower()
    category = (candidate.get("category") or "").lower()

    # Strip region/color suffixes: "galaxy s24 ultra 256gb black" → "galaxy s24 ultra"
    model = re.sub(r"\b(\d+\s*gb|black|white|blue|green|red|gold|silver|global|india|pro\s*max)\b", "", model)
    model = re.sub(r"\s+", " ", model).strip()

    return f"{brand}|{model}|{category}"


def _merge_variants(a: list[dict], b: list[dict]) -> list[dict]:
    """Merge two variant lists, deduplicating by (ram_gb, storage_gb, vendor)."""
    seen: set[tuple] = set()
    merged: list[dict] = []
    for v in a + b:
        key = (v.get("ram_gb"), v.get("storage_gb"), v.get("vendor"))
        if key not in seen:
            seen.add(key)
            merged.append(v)
    return merged


async def deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate candidate products by brand + model + category.

    Preserves the candidate with the lowest product_id as canonical.
    Merges variants from duplicates to ensure all prices are available.
    Preserves original vector ranking order.
    """
    if not candidates:
        return []

    canonical_map: dict[str, dict[str, Any]] = {}
    order: list[str] = []  # Track insertion order for ranking preservation

    for candidate in candidates:
        key = _canonical_key(candidate)

        if key not in canonical_map:
            canonical_map[key] = dict(candidate)  # shallow copy
            order.append(key)
        else:
            # Merge: keep lower product_id as canonical, merge variants
            existing = canonical_map[key]
            if candidate.get("product_id", 9999999) < existing.get("product_id", 9999999):
                # New candidate is more canonical — swap but keep merged variants
                merged_variants = _merge_variants(
                    candidate.get("variants", []),
                    existing.get("variants", []),
                )
                canonical_map[key] = dict(candidate)
                canonical_map[key]["variants"] = merged_variants
            else:
                # Existing is canonical — just merge variants
                canonical_map[key]["variants"] = _merge_variants(
                    existing.get("variants", []),
                    candidate.get("variants", []),
                )

    deduped = [canonical_map[k] for k in order]

    if len(deduped) < len(candidates):
        logger.info(
            "candidates_deduplicated",
            before=len(candidates),
            after=len(deduped),
        )

    return deduped
