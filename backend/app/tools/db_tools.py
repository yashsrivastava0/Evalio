"""Database tool functions shared across agents.

All DB read operations live here. Agents call these; they don't write
raw SQLAlchemy queries themselves.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.tables import Product, ReviewSummary

logger = get_logger(__name__)


async def fetch_product_specs(db: AsyncSession, product_id: int) -> dict[str, Any]:
    """Fetch canonical_title, brand, category, and specs_jsonb for a product."""
    result = await db.execute(
        select(
            Product.id,
            Product.product_slug,
            Product.canonical_title,
            Product.brand,
            Product.category,
            Product.specs_jsonb,
        ).where(Product.id == product_id)
    )
    row = result.first()
    if not row:
        return {}
    return {
        "product_id": row[0],
        "product_slug": row[1],
        "canonical_title": row[2],
        "brand": row[3],
        "category": row[4].value if hasattr(row[4], "value") else str(row[4]),
        "specs_jsonb": row[5] or {},
    }


async def fetch_review_summaries(
    db: AsyncSession,
    product_ids: list[int],
) -> list[dict[str, Any]]:
    """Fetch ReviewSummary rows for a list of product IDs.

    Returns empty list gracefully if no reviews exist — agents must
    handle missing review data.
    """
    if not product_ids:
        return []

    result = await db.execute(
        select(
            ReviewSummary.product_id,
            ReviewSummary.summary_text,
            ReviewSummary.sentiment_score,
            ReviewSummary.source_count,
            ReviewSummary.last_synced,
        ).where(ReviewSummary.product_id.in_(product_ids))
    )
    rows = result.fetchall()

    return [
        {
            "product_id": row[0],
            "summary_text": row[1] or "",
            "sentiment_score": row[2],
            "source_count": row[3] or 0,
            "last_synced": row[4].isoformat() if row[4] else None,
        }
        for row in rows
    ]


async def fetch_products_by_slugs(
    db: AsyncSession,
    slugs: list[str],
) -> list[dict[str, Any]]:
    """Fetch minimal product data for a list of slugs (for validation)."""
    if not slugs:
        return []

    result = await db.execute(
        select(Product.id, Product.product_slug, Product.canonical_title, Product.brand)
        .where(Product.product_slug.in_(slugs))
    )
    rows = result.fetchall()
    return [
        {"product_id": row[0], "product_slug": row[1], "canonical_title": row[2], "brand": row[3]}
        for row in rows
    ]
