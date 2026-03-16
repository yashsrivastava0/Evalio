"""Seed demo data — 8 smartphones with variants, scores, and embeddings.

Usage (from backend/ directory):
    python scripts/seed_demo.py

Requirements:
    - Docker stack running (postgres on localhost:5433)
    - OPENAI_API_KEY set in backend/.env
    - Migrations at head (alembic upgrade head)

Idempotent: products are inserted with ON CONFLICT (product_slug) DO NOTHING.
Re-running will skip existing products but still attempt embeddings for any
products that don't have one yet.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── ensure backend/ is on sys.path when run as a script ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.models.database import async_session_factory
from app.models.tables import (
    ComponentScore,
    Embedding,
    EmbeddingType,
    Product,
    ProductCategory,
    Variant,
)
from app.core.constants import EMBEDDING_VERSION
from app.services.embedding import create_embedding_provider

# ── Demo product definitions ───────────────────────────────────────────────────

SMARTPHONES = [
    {
        "product_slug": "samsung-galaxy-a55-5g",
        "brand": "Samsung",
        "model_name": "Galaxy A55 5G",
        "canonical_title": "Samsung Galaxy A55 5G 8GB 128GB",
        "specs_jsonb": {
            "processor": "Exynos 1480",
            "ram_gb": 8,
            "storage_gb": 128,
            "battery_mah": 5000,
            "camera_mp": 50,
            "display_hz": 120,
            "display_inches": 6.6,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 128, "color": "Awesome Navy", "price_inr": 34999, "vendor": "Samsung"},
            {"ram_gb": 8, "storage_gb": 256, "color": "Awesome Lilac", "price_inr": 37999, "vendor": "Samsung"},
        ],
        "component_scores": {
            "performance_score": 0.72,
            "battery_score": 0.75,
            "camera_score": 0.78,
            "display_score": 0.80,
            "build_score": 0.82,
            "price_score": 0.62,
            "popularity_score": 0.85,
        },
    },
    {
        "product_slug": "oneplus-nord-ce-4",
        "brand": "OnePlus",
        "model_name": "Nord CE 4",
        "canonical_title": "OnePlus Nord CE 4 8GB 128GB",
        "specs_jsonb": {
            "processor": "Snapdragon 7s Gen 2",
            "ram_gb": 8,
            "storage_gb": 128,
            "battery_mah": 5500,
            "camera_mp": 50,
            "display_hz": 120,
            "display_inches": 6.7,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 128, "color": "Dark Chrome", "price_inr": 24999, "vendor": "OnePlus"},
            {"ram_gb": 12, "storage_gb": 256, "color": "Celadon Marble", "price_inr": 28999, "vendor": "OnePlus"},
        ],
        "component_scores": {
            "performance_score": 0.78,
            "battery_score": 0.82,
            "camera_score": 0.70,
            "display_score": 0.78,
            "build_score": 0.75,
            "price_score": 0.78,
            "popularity_score": 0.80,
        },
    },
    {
        "product_slug": "xiaomi-redmi-note-13-pro-plus",
        "brand": "Xiaomi",
        "model_name": "Redmi Note 13 Pro+",
        "canonical_title": "Xiaomi Redmi Note 13 Pro+ 8GB 256GB",
        "specs_jsonb": {
            "processor": "MediaTek Dimensity 7200 Ultra",
            "ram_gb": 8,
            "storage_gb": 256,
            "battery_mah": 5000,
            "camera_mp": 200,
            "display_hz": 120,
            "display_inches": 6.67,
            "os": "Android 13",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 256, "color": "Midnight Black", "price_inr": 26999, "vendor": "Xiaomi"},
            {"ram_gb": 12, "storage_gb": 256, "color": "Aurora Purple", "price_inr": 29999, "vendor": "Xiaomi"},
        ],
        "component_scores": {
            "performance_score": 0.74,
            "battery_score": 0.76,
            "camera_score": 0.90,
            "display_score": 0.82,
            "build_score": 0.78,
            "price_score": 0.75,
            "popularity_score": 0.82,
        },
    },
    {
        "product_slug": "realme-gt-6t",
        "brand": "Realme",
        "model_name": "GT 6T",
        "canonical_title": "Realme GT 6T 8GB 256GB",
        "specs_jsonb": {
            "processor": "Snapdragon 7s Gen 3",
            "ram_gb": 8,
            "storage_gb": 256,
            "battery_mah": 5500,
            "camera_mp": 50,
            "display_hz": 144,
            "display_inches": 6.78,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 256, "color": "Fluid Silver", "price_inr": 29999, "vendor": "Realme"},
            {"ram_gb": 12, "storage_gb": 256, "color": "Razor Green", "price_inr": 32999, "vendor": "Realme"},
        ],
        "component_scores": {
            "performance_score": 0.82,
            "battery_score": 0.80,
            "camera_score": 0.72,
            "display_score": 0.85,
            "build_score": 0.74,
            "price_score": 0.72,
            "popularity_score": 0.76,
        },
    },
    {
        "product_slug": "iqoo-z9-5g",
        "brand": "iQOO",
        "model_name": "Z9 5G",
        "canonical_title": "iQOO Z9 5G 8GB 128GB",
        "specs_jsonb": {
            "processor": "MediaTek Dimensity 7200",
            "ram_gb": 8,
            "storage_gb": 128,
            "battery_mah": 6000,
            "camera_mp": 50,
            "display_hz": 120,
            "display_inches": 6.78,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 128, "color": "Storm Black", "price_inr": 19999, "vendor": "iQOO"},
            {"ram_gb": 8, "storage_gb": 256, "color": "Brushed Green", "price_inr": 21999, "vendor": "iQOO"},
        ],
        "component_scores": {
            "performance_score": 0.80,
            "battery_score": 0.88,
            "camera_score": 0.68,
            "display_score": 0.76,
            "build_score": 0.72,
            "price_score": 0.88,
            "popularity_score": 0.74,
        },
    },
    {
        "product_slug": "motorola-edge-50-pro",
        "brand": "Motorola",
        "model_name": "Edge 50 Pro",
        "canonical_title": "Motorola Edge 50 Pro 12GB 256GB",
        "specs_jsonb": {
            "processor": "Snapdragon 7 Gen 3",
            "ram_gb": 12,
            "storage_gb": 256,
            "battery_mah": 4500,
            "camera_mp": 50,
            "display_hz": 144,
            "display_inches": 6.7,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 12, "storage_gb": 256, "color": "Black Beauty", "price_inr": 31999, "vendor": "Motorola"},
            {"ram_gb": 12, "storage_gb": 256, "color": "Luxe Lavender", "price_inr": 31999, "vendor": "Motorola"},
        ],
        "component_scores": {
            "performance_score": 0.79,
            "battery_score": 0.68,
            "camera_score": 0.82,
            "display_score": 0.86,
            "build_score": 0.84,
            "price_score": 0.68,
            "popularity_score": 0.72,
        },
    },
    {
        "product_slug": "nothing-phone-2a",
        "brand": "Nothing",
        "model_name": "Phone (2a)",
        "canonical_title": "Nothing Phone (2a) 8GB 128GB",
        "specs_jsonb": {
            "processor": "MediaTek Dimensity 7200 Pro",
            "ram_gb": 8,
            "storage_gb": 128,
            "battery_mah": 5000,
            "camera_mp": 50,
            "display_hz": 120,
            "display_inches": 6.7,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 128, "color": "Black", "price_inr": 23999, "vendor": "Nothing"},
            {"ram_gb": 12, "storage_gb": 256, "color": "White", "price_inr": 27999, "vendor": "Nothing"},
        ],
        "component_scores": {
            "performance_score": 0.75,
            "battery_score": 0.74,
            "camera_score": 0.71,
            "display_score": 0.77,
            "build_score": 0.86,
            "price_score": 0.80,
            "popularity_score": 0.78,
        },
    },
    {
        "product_slug": "vivo-v30e",
        "brand": "Vivo",
        "model_name": "V30e",
        "canonical_title": "Vivo V30e 8GB 128GB",
        "specs_jsonb": {
            "processor": "Snapdragon 6 Gen 1",
            "ram_gb": 8,
            "storage_gb": 128,
            "battery_mah": 5000,
            "camera_mp": 64,
            "display_hz": 120,
            "display_inches": 6.67,
            "os": "Android 14",
            "5g": True,
        },
        "variants": [
            {"ram_gb": 8, "storage_gb": 128, "color": "Peacock Green", "price_inr": 21999, "vendor": "Vivo"},
            {"ram_gb": 8, "storage_gb": 256, "color": "Silk Red", "price_inr": 23999, "vendor": "Vivo"},
        ],
        "component_scores": {
            "performance_score": 0.70,
            "battery_score": 0.73,
            "camera_score": 0.82,
            "display_score": 0.75,
            "build_score": 0.76,
            "price_score": 0.82,
            "popularity_score": 0.70,
        },
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_specs_hash(specs_jsonb: dict) -> str:
    """Canonical SHA-256 of specs dict — must match embedding_pipeline logic."""
    canonical = json.dumps(specs_jsonb, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _compute_embedding_hash(vector: list[float]) -> str:
    """SHA-256 of embedding bytes (first 16 hex chars)."""
    raw = json.dumps(vector, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _specs_to_text(specs_jsonb: dict, canonical_title: str) -> str:
    """Match exact format used by embedding_pipeline._specs_to_text()."""
    parts = [f"{k}: {v}" for k, v in sorted(specs_jsonb.items()) if v is not None]
    spec_text = " | ".join(parts)
    return f"{canonical_title} | {spec_text}" if canonical_title else spec_text


# ── Main seed function ─────────────────────────────────────────────────────────


async def seed() -> None:
    settings = get_settings()
    provider = create_embedding_provider()
    if provider is None:
        print("  ❌ No embedding provider available. Check OPENAI_API_KEY in .env")
        sys.exit(1)

    print(f"  Embedding provider: {settings.embedding_model} (dim={provider.get_dimension()})")

    # Validate dimension matches schema
    expected_dim = getattr(settings, "embedding_dim", 1536)
    if provider.get_dimension() != expected_dim:
        print(
            f"  ❌ Dimension mismatch: provider={provider.get_dimension()}, "
            f"schema expects {expected_dim}. Update settings.embedding_dim."
        )
        sys.exit(1)

    products_seeded = 0
    variants_seeded = 0
    scores_seeded = 0
    embeddings_seeded = 0

    async with async_session_factory() as db:
        now = datetime.now(timezone.utc)

        for p in SMARTPHONES:
            # ── 1. Upsert product (ON CONFLICT DO NOTHING) ──
            stmt = (
                pg_insert(Product)
                .values(
                    product_slug=p["product_slug"],
                    brand=p["brand"],
                    model_name=p["model_name"],
                    category=ProductCategory.smartphone,
                    specs_jsonb=p["specs_jsonb"],
                    canonical_title=p["canonical_title"],
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=["product_slug"])
                .returning(Product.id)
            )
            result = await db.execute(stmt)
            row = result.fetchone()

            if row is None:
                # Product already existed — fetch its id
                existing = await db.execute(
                    select(Product.id).where(Product.product_slug == p["product_slug"])
                )
                product_id = existing.scalar_one()
            else:
                product_id = row[0]
                products_seeded += 1

            # ── 2. Upsert variants ──
            for v in p["variants"]:
                vstmt = (
                    pg_insert(Variant)
                    .values(
                        product_id=product_id,
                        ram_gb=v["ram_gb"],
                        storage_gb=v["storage_gb"],
                        color=v["color"],
                        price_inr=float(v["price_inr"]),
                        currency="INR",
                        availability_bool=True,
                        last_price_update=now,
                        vendor=v["vendor"],
                    )
                    .on_conflict_do_nothing()
                )
                vresult = await db.execute(vstmt)
                variants_seeded += vresult.rowcount

            # ── 3. Upsert component scores ──
            cstmt = (
                pg_insert(ComponentScore)
                .values(
                    product_id=product_id,
                    computed_at=now,
                    score_version=1,
                    **p["component_scores"],
                )
                .on_conflict_do_nothing()
            )
            cresult = await db.execute(cstmt)
            scores_seeded += cresult.rowcount

            await db.commit()

            # ── 4. Generate and upsert embedding ──
            existing_emb = await db.execute(
                select(Embedding.id).where(
                    Embedding.product_id == product_id,
                    Embedding.embedding_type == EmbeddingType.specs,
                )
            )
            if existing_emb.scalar_one_or_none() is not None:
                print(f"  -> {p['canonical_title']}: embedding already exists, skipping")
                continue

            embed_text = _specs_to_text(p["specs_jsonb"], p["canonical_title"])
            print(f"  -> Embedding: {p['canonical_title']}")
            vector = await provider.embed(embed_text)

            if len(vector) != expected_dim:
                print(
                    f"  ❌ Got dimension {len(vector)} for {p['product_slug']}, "
                    f"expected {expected_dim}. Aborting."
                )
                await db.rollback()
                sys.exit(1)

            embedding_hash = _compute_embedding_hash(vector)
            specs_hash = _compute_specs_hash(p["specs_jsonb"])

            emb_stmt = (
                pg_insert(Embedding)
                .values(
                    product_id=product_id,
                    embedding_vector=vector,
                    embedding_type=EmbeddingType.specs,
                    embedding_hash=embedding_hash,
                    embedding_version=EMBEDDING_VERSION,
                    specs_hash=specs_hash,
                    reviews_hash=None,
                    created_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["product_id", "embedding_type"],
                    set_={
                        "embedding_vector": vector,
                        "embedding_hash": embedding_hash,
                        "specs_hash": specs_hash,
                        "embedding_version": EMBEDDING_VERSION,
                    },
                )
            )
            await db.execute(emb_stmt)
            await db.commit()
            embeddings_seeded += 1

    print(
        f"\n  Seeded {products_seeded} products, {variants_seeded} variants, "
        f"{scores_seeded} component_scores, {embeddings_seeded} embeddings"
    )


if __name__ == "__main__":
    asyncio.run(seed())
