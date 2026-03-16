"""Core fetch pipeline — two-tier scraping orchestrator.

Architecture:
  Fetch Engine
     ├─ Tier-1 Structured Scraper (runs FIRST)
     │      Smartprix  → search_product() → scrape()
     │      91mobiles  → search_product() → scrape()  (fallback if Smartprix fails)
     │
     └─ Tier-2 Internet Discovery Agent (runs ONLY if Tier-1 fails / data incomplete)
            Search DuckDuckGo → filter to ALLOWED_DOMAINS → scrape → normalize

Normalization pipeline (shared by both tiers):
    raw HTML → structured extraction → Pydantic schema validation
             → normalized JSON → DB insertion
    Then triggers: embedding generation → pgvector indexing

Architectural rules:
  1. Lock acquired FIRST. If already locked → exit silently (idempotent).
  2. category passed in — NEVER inferred by scraper.
  3. Tier-1 runs first. Tier-2 runs ONLY if Tier-1 fails or data is incomplete.
  4. Circuit breaker: if extraction sparse → abort before any DB write.
  5. Recompute TTL: skip score recompute if scores are fresh within 30 days
     and score_version matches current SCORE_VERSION.
  6. Embed lock prevents duplicate embedding calls.
  7. Release lock in finally block — always, even on failure.

Import strategy:
  All cross-package imports (app.*, scraper.*) are lazy (inside function bodies)
  to work correctly with the runtime sys.path manipulation below.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── Runtime path setup ────────────────────────────────────────────────────────
# Needed so that 'app.*' (backend) and 'scraper.*' resolve at runtime.
# All cross-package imports below are lazy (inside function bodies).

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_backend_path = os.path.join(_repo_root, "backend")
for _p in (_backend_path, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────

SCORE_RECOMPUTE_TTL_DAYS: int = 30


# ── Small utilities ───────────────────────────────────────────────────────────


def _raw_is_usable(raw: Any) -> bool:
    """Return True if raw scrape result is non-None and error-free."""
    if raw is None:
        return False
    return not getattr(raw, "error", None)


def _log_scraper_error(
    product_slug: str,
    source: str,
    error_type: str,
    detail: str = "",
) -> None:
    """Emit structured error log."""
    logger.error(
        "scraper_error",
        extra={
            "product_slug": product_slug,
            "source": source,
            "error_type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        },
    )


# ── Tier-1 orchestration ──────────────────────────────────────────────────────


async def _run_tier1(product_name: str, category: str) -> Any:
    """Run Tier-1 structured catalog scrapers (Smartprix → 91mobiles) in order.

    Returns the first successful RawProductData, or None if all fail.
    """
    from scraper.playwright_scraper.smartprix_scraper import SmartprixScraper
    from scraper.playwright_scraper.ninety1mobiles_scraper import NinetyOneMobilesScraper

    tier1_scrapers = [SmartprixScraper, NinetyOneMobilesScraper]

    for ScraperCls in tier1_scrapers:
        scraper = ScraperCls(category=category)
        scraper_name = ScraperCls.__name__
        try:
            logger.info(
                "tier1_attempt",
                extra={"scraper": scraper_name, "product": product_name},
            )
            url = await scraper.search_product(product_name, category)
            if not url:
                logger.info(
                    "tier1_search_no_result",
                    extra={"scraper": scraper_name, "product": product_name},
                )
                continue

            raw = await scraper._scrape_with_retry(url)
            if not _raw_is_usable(raw):
                logger.warning(
                    "tier1_scrape_unusable",
                    extra={"scraper": scraper_name, "url": url},
                )
                continue

            if not scraper._validate_extraction(raw):
                logger.warning(
                    "tier1_sparse_specs",
                    extra={"scraper": scraper_name, "spec_count": len(raw.specs)},
                )
                continue

            logger.info(
                "tier1_success",
                extra={"scraper": scraper_name, "spec_count": len(raw.specs), "url": url},
            )
            return raw

        except Exception as exc:
            logger.warning(
                "tier1_exception",
                extra={"scraper": scraper_name, "error": str(exc)},
            )
            continue

    logger.warning("tier1_all_failed", extra={"product": product_name})
    return None


# ── Tier-2 orchestration ──────────────────────────────────────────────────────


async def _run_tier2(product_name: str, category: str) -> Any:
    """Run Tier-2 Internet Discovery Agent (ALLOWED_DOMAINS whitelist only).

    Activated only when Tier-1 fails or returns incomplete data.
    """
    from scraper.discovery.discovery_agent import DiscoveryAgent

    logger.info("tier2_attempt", extra={"product": product_name, "category": category})
    try:
        agent = DiscoveryAgent(category=category)
        raw = await agent.discover(product_name)
        if _raw_is_usable(raw):
            logger.info(
                "tier2_success",
                extra={"spec_count": len(raw.specs), "url": raw.url},
            )
        return raw
    except Exception as exc:
        logger.error("tier2_exception", extra={"error": str(exc)})
        return None


# ── DB helpers ────────────────────────────────────────────────────────────────


async def _upsert_product(db_session: Any, product_dict: dict) -> int:
    """Upsert product record. Returns product_id."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.tables import Product

    stmt = pg_insert(Product).values(**product_dict)
    stmt = stmt.on_conflict_do_update(
        index_elements=["product_slug"],
        set_={
            "brand": stmt.excluded.brand,
            "model_name": stmt.excluded.model_name,
            "canonical_title": stmt.excluded.canonical_title,
            "category": stmt.excluded.category,
            "release_date": stmt.excluded.release_date,
            "specs_jsonb": stmt.excluded.specs_jsonb,
            "updated_at": datetime.now(timezone.utc),
        },
    )
    stmt = stmt.returning(Product.id)
    result = await db_session.execute(stmt)
    await db_session.flush()
    return result.scalar_one()


async def _upsert_variant(db_session: Any, variant_dict: dict) -> int:
    """Upsert variant record. Returns variant_id."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.tables import Variant

    stmt = pg_insert(Variant).values(**variant_dict)
    stmt = stmt.on_conflict_do_update(
        index_elements=["product_id", "vendor", "ram_gb", "storage_gb"],
        set_={
            "price_inr": stmt.excluded.price_inr,
            "availability_bool": stmt.excluded.availability_bool,
            "last_price_update": stmt.excluded.last_price_update,
            "affiliate_url": stmt.excluded.affiliate_url,
        },
    )
    stmt = stmt.returning(Variant.id)
    result = await db_session.execute(stmt)
    await db_session.flush()
    return result.scalar_one()


async def _upsert_price_history(
    db_session: Any, variant_id: int, price_inr: float, vendor: str
) -> None:
    """Always insert a price_history row (historical log, never upsert)."""
    from app.models.tables import PriceHistory

    row = PriceHistory(
        variant_id=variant_id,
        price_inr=price_inr,
        vendor=vendor,
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(row)
    await db_session.flush()


async def _upsert_component_scores(
    db_session: Any, product_id: int, scores: Any
) -> None:
    """Upsert component scores for a product."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.tables import ComponentScore

    stmt = pg_insert(ComponentScore).values(
        product_id=product_id,
        performance_score=scores.performance_score,
        battery_score=scores.battery_score,
        camera_score=scores.camera_score,
        display_score=scores.display_score,
        build_score=scores.build_score,
        price_score=scores.price_score,
        popularity_score=scores.popularity_score,
        score_version=scores.score_version,
        computed_at=scores.computed_at,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["product_id"],
        set_={
            "performance_score": stmt.excluded.performance_score,
            "battery_score": stmt.excluded.battery_score,
            "camera_score": stmt.excluded.camera_score,
            "display_score": stmt.excluded.display_score,
            "build_score": stmt.excluded.build_score,
            "price_score": stmt.excluded.price_score,
            "popularity_score": stmt.excluded.popularity_score,
            "score_version": stmt.excluded.score_version,
            "computed_at": stmt.excluded.computed_at,
        },
    )
    await db_session.execute(stmt)
    await db_session.flush()


async def _scores_are_fresh(db_session: Any, product_id: int) -> bool:
    """Return True if existing component scores are fresh (no recompute needed).

    Checks:
      - score_version matches current SCORE_VERSION constant
      - computed_at is within SCORE_RECOMPUTE_TTL_DAYS
    """
    from sqlalchemy import select
    from app.models.tables import ComponentScore
    from app.domain.scoring_config import SCORE_VERSION

    result = await db_session.execute(
        select(ComponentScore).where(ComponentScore.product_id == product_id)
    )
    existing = result.scalar_one_or_none()
    if existing is None:
        return False

    if existing.score_version != SCORE_VERSION:
        return False

    computed_at = existing.computed_at
    if computed_at is None:
        return False

    age = datetime.now(timezone.utc) - computed_at
    return age < timedelta(days=SCORE_RECOMPUTE_TTL_DAYS)


# ── Main pipeline ─────────────────────────────────────────────────────────────


async def run_fetch_task(
    product_slug: str,
    category: str,
    db_session: Any,
    redis_client: Any,
    *,
    url: str | None = None,
    vendor: str | None = None,
    product_name: str | None = None,
) -> dict:
    """Full two-tier fetch pipeline: Tier-1 → Tier-2 → normalize → upsert → score → embed.

    Tier-1 (structured catalog scrapers) runs first — Smartprix, then 91mobiles.
    Tier-2 (internet discovery agent) runs ONLY if Tier-1 fails.
    Both tiers feed the same normalization pipeline.

    Args:
        product_slug: Pre-computed collision-safe slug.
        category: "smartphone" | "laptop" — never inferred.
        db_session: Async SQLAlchemy session.
        redis_client: Async Redis client.
        url: Optional direct URL hint (legacy / refresh path).
        vendor: Optional vendor hint (legacy compatibility).
        product_name: Human-readable name for search. Defaults to slug → spaces.

    Returns:
        Result dict with status, product_id, slug, tier, and timing metadata.
    """
    from app.services.cache import (
        acquire_scrape_lock,
        release_scrape_lock,
        set_product_fetch_status,
        acquire_embed_lock,
        release_embed_lock,
    )
    from scraper.normalization.spec_normalizer import normalize_specs
    from scraper.normalization.variant_builder import build_product_dict, build_variant_dict
    from scraper.scoring.component_score_calculator import compute_all_scores
    from scraper.playwright_scraper.base_scraper import BaseScraper

    start_ts = datetime.now(timezone.utc)
    effective_name = product_name or product_slug.replace("-", " ")

    # ── Step 1: Distributed lock ──────────────────────────────────────────────
    lock_acquired = await acquire_scrape_lock(redis_client, product_slug)
    if not lock_acquired:
        logger.info("scrape_skipped_locked", extra={"slug": product_slug})
        return {"status": "skipped", "reason": "already_locked", "slug": product_slug}

    await set_product_fetch_status(redis_client, product_slug, "fetching")

    # tier_used defined before try so it's always available in the except block
    tier_used: str = "tier1"

    try:
        # ── Step 2: Tier-1 — Structured Catalog Scrapers ──────────────────────
        tier_used = "tier1"
        t_scrape_start = datetime.now(timezone.utc)
        raw: Any = await _run_tier1(effective_name, category)

        # ── Step 3: Tier-2 — Internet Discovery Agent (fallback only) ─────────
        if not _raw_is_usable(raw):
            logger.info(
                "tier1_failed_activating_tier2",
                extra={"product": effective_name, "category": category},
            )
            tier_used = "tier2"
            raw = await _run_tier2(effective_name, category)

        scrape_ms = (datetime.now(timezone.utc) - t_scrape_start).total_seconds() * 1000

        # ── Verify we have usable data ────────────────────────────────────────
        if not _raw_is_usable(raw):
            error_msg = getattr(raw, "error", "no data returned") if raw is not None else "no data returned"
            raise RuntimeError(f"All scraping tiers failed for '{effective_name}': {error_msg}")

        # ── Step 4: Circuit breaker — sparse spec check ────────────────────────
        if not BaseScraper._validate_extraction(raw):
            valid_count = len([v for v in raw.specs.values() if v])
            _log_scraper_error(product_slug, tier_used, "sparse_specs",
                               f"Only {valid_count} valid spec fields")
            raise RuntimeError("circuit_breaker: sparse_specs")

        # ── Step 5: Normalize specs (same pipeline for both tiers) ─────────────
        t_norm_start = datetime.now(timezone.utc)
        specs = normalize_specs(raw.specs)
        norm_ms = (datetime.now(timezone.utc) - t_norm_start).total_seconds() * 1000

        # ── Step 6: Build DB dicts ────────────────────────────────────────────
        product_dict = build_product_dict(raw, specs, category)
        final_slug: str = product_dict["product_slug"]

        # ── Step 7: DB upsert (product → variant → price history) ─────────────
        t_insert_start = datetime.now(timezone.utc)
        product_id: int = await _upsert_product(db_session, product_dict)

        effective_vendor: str = getattr(raw, "vendor", None) or vendor or tier_used
        variant_dict = build_variant_dict(raw, product_id, effective_vendor)
        variant_id: int = await _upsert_variant(db_session, variant_dict)

        raw_price = getattr(raw, "price", None)
        if raw_price is not None:
            await _upsert_price_history(db_session, variant_id, raw_price, effective_vendor)

        insert_ms = (datetime.now(timezone.utc) - t_insert_start).total_seconds() * 1000

        # ── Step 8: Recompute scores if stale ─────────────────────────────────
        t_score_start = datetime.now(timezone.utc)
        if await _scores_are_fresh(db_session, product_id):
            logger.info("score_recompute_skipped_fresh", extra={"product_id": product_id})
            score_ms = 0.0
        else:
            scores = compute_all_scores(
                specs=specs,
                price_inr=getattr(raw, "price", None),
                review_count=getattr(raw, "review_count", 0),
                category=category,
                budget_max=None,
                sentiment_score=None,
            )
            await _upsert_component_scores(db_session, product_id, scores)
            score_ms = (datetime.now(timezone.utc) - t_score_start).total_seconds() * 1000

        await db_session.commit()

        # ── Step 9: Trigger embedding (fire-and-forget) ───────────────────────
        embed_acquired = await acquire_embed_lock(redis_client, product_id)
        if embed_acquired:
            logger.info(
                "embedding_trigger",
                extra={"product_id": product_id, "slug": final_slug, "tier": tier_used},
            )
        else:
            logger.info("embed_skipped_locked", extra={"product_id": product_id})

        # ── Step 10: Mark ready ───────────────────────────────────────────────
        await set_product_fetch_status(redis_client, product_slug, "ready")

        total_ms = (datetime.now(timezone.utc) - start_ts).total_seconds() * 1000
        logger.info(
            "fetch_task_complete",
            extra={
                "slug": final_slug, "product_id": product_id, "tier": tier_used,
                "scrape_ms": round(scrape_ms), "norm_ms": round(norm_ms),
                "insert_ms": round(insert_ms), "score_ms": round(score_ms),
                "total_ms": round(total_ms),
            },
        )

        return {
            "status": "ready",
            "product_id": product_id,
            "product_slug": final_slug,
            "tier": tier_used,
            "vendor": effective_vendor,
            "scrape_ms": round(scrape_ms),
            "norm_ms": round(norm_ms),
            "insert_ms": round(insert_ms),
            "score_ms": round(score_ms),
            "total_ms": round(total_ms),
        }

    except Exception as exc:
        _log_scraper_error(product_slug, tier_used, type(exc).__name__, str(exc))
        await set_product_fetch_status(redis_client, product_slug, "failed")
        try:
            await db_session.rollback()
        except Exception:
            pass
        return {
            "status": "failed",
            "product_slug": product_slug,
            "error": str(exc),
        }

    finally:
        from app.services.cache import release_scrape_lock
        await release_scrape_lock(redis_client, product_slug)
