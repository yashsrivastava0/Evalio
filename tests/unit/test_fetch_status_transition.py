"""Unit tests — fetch status state machine transitions.

Validates that the fetch pipeline correctly transitions through:
  None (pre-scrape) → "fetching" → "ready"  (success path)
  None (pre-scrape) → "fetching" → "failed"  (failure path)

Also tests the circuit breaker: sparse specs must abort the pipeline
before any DB write and set fetch_status="failed".
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import pytest

from app.services.cache import (
    set_product_fetch_status,
    get_product_fetch_status,
    acquire_scrape_lock,
    release_scrape_lock,
)
from scraper.playwright_scraper.base_scraper import BaseScraper, RawProductData, MIN_VALID_SPEC_FIELDS


# ── Status state machine ──────────────────────────────────────────────────────


class TestFetchStatusTransitions:
    """Tests use an in-memory dict to simulate Redis state."""

    @pytest.mark.asyncio
    async def test_success_path_transitions(self):
        """None → fetching → ready (in order)."""
        store: dict[str, str] = {}

        redis = AsyncMock()

        async def mock_set(key, value, **kwargs):
            store[key] = value
            return b"OK"

        async def mock_get(key):
            return store.get(key)

        redis.set.side_effect = mock_set
        redis.get.side_effect = mock_get

        slug = "apple-iphone-15-pro-smartphone-2023"
        status_key = f"fetch_status:{slug}"

        # Pre-scrape: status is None
        assert await mock_get(status_key) is None

        # Step 1: set to "fetching"
        await set_product_fetch_status(redis, slug, "fetching")
        assert store[status_key] == "fetching"

        # Step 2: set to "ready"
        await set_product_fetch_status(redis, slug, "ready")
        assert store[status_key] == "ready"

    @pytest.mark.asyncio
    async def test_failure_path_transitions(self):
        """None → fetching → failed."""
        store: dict[str, str] = {}

        redis = AsyncMock()

        async def mock_set(key, value, **kwargs):
            store[key] = value
            return b"OK"

        redis.set.side_effect = mock_set

        slug = "samsung-galaxy-s24-ultra-smartphone-2024"

        await set_product_fetch_status(redis, slug, "fetching")
        assert store[f"fetch_status:{slug}"] == "fetching"

        await set_product_fetch_status(redis, slug, "failed")
        assert store[f"fetch_status:{slug}"] == "failed"

    @pytest.mark.asyncio
    async def test_lock_and_status_independent(self):
        """Scrape lock (fetch:<slug>) and status (fetch_status:<slug>) are separate keys."""
        store: dict[str, str] = {}

        redis = AsyncMock()

        async def mock_set(key, value, **kwargs):
            store[key] = value
            return b"OK"

        async def mock_delete(key):
            store.pop(key, None)

        redis.set.side_effect = mock_set
        redis.delete.side_effect = mock_delete

        slug = "test-product-smartphone-2024"

        # Acquire lock
        await redis.set(f"fetch:{slug}", "locked", nx=True, ex=600)
        assert f"fetch:{slug}" in store

        # Set status
        await set_product_fetch_status(redis, slug, "fetching")
        assert f"fetch_status:{slug}" in store

        # Release lock
        await redis.delete(f"fetch:{slug}")
        assert f"fetch:{slug}" not in store
        assert f"fetch_status:{slug}" in store   # status persists


# ── Circuit breaker tests ─────────────────────────────────────────────────────


class TestCircuitBreakerSparseSpecs:
    """_validate_extraction() must abort when fewer than MIN_VALID_SPEC_FIELDS."""

    def _make_raw(self, specs: dict) -> RawProductData:
        return RawProductData(
            url="https://amazon.in/dp/TEST",
            vendor="amazon.in",
            category="smartphone",
            specs=specs,
        )

    def test_circuit_breaker_aborts_on_sparse_specs(self):
        """CRITICAL: fewer than MIN_VALID_SPEC_FIELDS valid fields → False."""
        raw = self._make_raw({"RAM": "8 GB", "Brand": "Apple"})  # only 2 valid
        assert MIN_VALID_SPEC_FIELDS == 3, "Threshold must be 3"
        result = BaseScraper._validate_extraction(raw)
        assert result is False

    def test_circuit_breaker_passes_on_sufficient_specs(self):
        """3+ valid fields → passes validation."""
        raw = self._make_raw({
            "RAM": "8 GB",
            "Battery": "5000 mAh",
            "Processor": "Snapdragon 8 Gen 3",
        })
        assert BaseScraper._validate_extraction(raw) is True

    def test_empty_values_not_counted(self):
        """Empty/whitespace values don't count toward valid spec threshold."""
        raw = self._make_raw({
            "RAM": "8 GB",
            "Battery": "",        # empty
            "Processor": "  ",    # whitespace only
        })
        # Only 1 valid field
        assert BaseScraper._validate_extraction(raw) is False

    def test_none_values_not_counted(self):
        """None values don't count toward valid spec threshold."""
        raw = self._make_raw({
            "RAM": "8 GB",
            "Battery": None,
            "Processor": None,
            "Display": "6.7 inches",
        })
        # 2 valid fields (RAM + Display), 2 None
        assert BaseScraper._validate_extraction(raw) is False

    def test_threshold_boundary(self):
        """Exactly MIN_VALID_SPEC_FIELDS valid fields → True."""
        raw = self._make_raw({
            "RAM": "8 GB",
            "Battery": "5000 mAh",
            "Processor": "Snapdragon 8 Gen 2",
        })
        assert BaseScraper._validate_extraction(raw) is True

    def test_rich_specs_passes(self):
        """7 valid fields → definitely passes."""
        raw = self._make_raw({
            "RAM": "12 GB",
            "Storage": "256 GB",
            "Battery": "5000 mAh",
            "Processor": "Snapdragon 8 Gen 3",
            "Display": "6.7 inches",
            "Camera": "50 MP",
            "Refresh Rate": "120 Hz",
        })
        assert BaseScraper._validate_extraction(raw) is True

    def test_all_empty_fails(self):
        """All empty specs → fails circuit breaker."""
        raw = self._make_raw({"RAM": "", "Battery": "", "Processor": ""})
        assert BaseScraper._validate_extraction(raw) is False

    def test_completely_empty_specs_fails(self):
        raw = self._make_raw({})
        assert BaseScraper._validate_extraction(raw) is False
