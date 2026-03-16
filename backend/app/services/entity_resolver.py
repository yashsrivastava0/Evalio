"""Entity Resolver Service.

Resolves vague or partial product references to canonical DB product slugs.

Examples:
    "s24"       → "samsung-galaxy-s24"
    "iphone 15" → "apple-iphone-15"
    "s24u"      → "samsung-galaxy-s24-ultra"

Strategy:
    1. Exact slug match
    2. Model name contains (case-insensitive)
    3. Canonical title contains (case-insensitive, token-based)
    4. Brand + partial model name match

Called by: Hybrid Retrieval Agent for COMPARE_PRODUCTS intent before SQL filter.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.tables import Product

logger = get_logger(__name__)


# ── Alias table for common shorthand names ────────────────────────────────────

_BRAND_ALIASES: dict[str, str] = {
    "samsung": "samsung",
    "apple": "apple",
    "iphone": "apple",
    "pixel": "google",
    "google": "google",
    "oneplus": "oneplus",
    "1+": "oneplus",
    "xiaomi": "xiaomi",
    "redmi": "xiaomi",
    "poco": "xiaomi",
    "realme": "realme",
    "oppo": "oppo",
    "vivo": "vivo",
    "motorola": "motorola",
    "moto": "motorola",
    "nokia": "nokia",
    "asus": "asus",
    "rog": "asus",
    "sony": "sony",
    "xperia": "sony",
    "dell": "dell",
    "hp": "hp",
    "lenovo": "lenovo",
    "acer": "acer",
    "lg": "lg",
}

_MODEL_SHORTHANDS: dict[str, str] = {
    "s24u": "galaxy s24 ultra",
    "s24+": "galaxy s24+",
    "s24 plus": "galaxy s24+",
    "s23u": "galaxy s23 ultra",
    "s23+": "galaxy s23+",
    "s22u": "galaxy s22 ultra",
    "iphone15": "iphone 15",
    "iphone 15pro": "iphone 15 pro",
    "ip15": "iphone 15",
}


def _normalize_entity(entity: str) -> str:
    """Normalize entity string for matching."""
    e = entity.lower().strip()
    # Expand known shorthands
    e = _MODEL_SHORTHANDS.get(e, e)
    # Remove extra spaces
    e = re.sub(r"\s+", " ", e).strip()
    return e


async def resolve_product_entities(
    entities: list[str],
    db: AsyncSession,
) -> list[str]:
    """Resolve a list of entity strings to canonical product slugs.

    Returns slugs in the same order as input entities.
    Unresolved entities produce no slug (skipped).
    """
    if not entities:
        return []

    resolved: list[str] = []
    seen: set[str] = set()

    for entity in entities:
        slug = await _resolve_single(entity, db)
        if slug and slug not in seen:
            resolved.append(slug)
            seen.add(slug)
        else:
            logger.info("entity_unresolved", entity=entity)

    logger.info("entity_resolution_complete", input=entities, output=resolved)
    return resolved


async def _resolve_single(entity: str, db: AsyncSession) -> str | None:
    """Try to resolve a single entity to a product_slug."""
    normalized = _normalize_entity(entity)
    slug_form = normalized.replace(" ", "-")

    # 1. Exact slug match
    result = await db.execute(
        select(Product.product_slug).where(Product.product_slug == slug_form)
    )
    row = result.first()
    if row:
        return row[0]

    # 2. Slug contains the normalized form (handles prefix matches)
    result = await db.execute(
        select(Product.product_slug).where(
            Product.product_slug.ilike(f"%{slug_form}%")
        ).limit(1)
    )
    row = result.first()
    if row:
        return row[0]

    # 3. Canonical title ILIKE
    result = await db.execute(
        select(Product.product_slug).where(
            Product.canonical_title.ilike(f"%{normalized}%")
        ).limit(1)
    )
    row = result.first()
    if row:
        return row[0]

    # 4. Model name contains (split tokens)
    tokens = [t for t in normalized.split() if len(t) > 1]
    if len(tokens) >= 2:
        conditions = [Product.model_name.ilike(f"%{t}%") for t in tokens]
        result = await db.execute(
            select(Product.product_slug)
            .where(or_(*conditions))
            .limit(1)
        )
        row = result.first()
        if row:
            return row[0]

    return None


async def resolve_to_product_ids(
    slugs: list[str],
    db: AsyncSession,
) -> list[int]:
    """Convert already-resolved slugs to product IDs."""
    if not slugs:
        return []

    result = await db.execute(
        select(Product.id, Product.product_slug).where(
            Product.product_slug.in_(slugs)
        )
    )
    rows = result.fetchall()
    # Preserve input order
    slug_to_id = {row[1]: row[0] for row in rows}
    return [slug_to_id[s] for s in slugs if s in slug_to_id]
