"""Unit tests — fetch locking (Redis distributed lock behavior).

Tests:
  - acquire_scrape_lock: atomic test-and-set, returns True (acquired) / False (locked)
  - release_scrape_lock: DEL removes the key
  - acquire_embed_lock / release_embed_lock: same pattern for embedding
  - Double-acquire by same caller returns False (idempotent)
  - set_product_fetch_status / get_product_fetch_status round-trip

All tests use a mock Redis client to avoid real Redis dependency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.cache import (
    acquire_scrape_lock,
    release_scrape_lock,
    set_product_fetch_status,
    get_product_fetch_status,
    acquire_embed_lock,
    release_embed_lock,
)


# ── Mock Redis ────────────────────────────────────────────────────────────────


def _make_redis(set_return=b"OK", get_return=None):
    """Build an async mock Redis client."""
    redis = AsyncMock()
    redis.set.return_value = set_return   # SET NX EX → None means already locked
    redis.get.return_value = get_return
    redis.delete.return_value = 1
    return redis


# ── Scrape lock tests ─────────────────────────────────────────────────────────


class TestAcquireScrapeLock:
    @pytest.mark.asyncio
    async def test_acquire_returns_true_when_free(self):
        redis = _make_redis(set_return=b"OK")   # SET NX returns "OK" on success
        acquired = await acquire_scrape_lock(redis, "apple-iphone-15-pro-smartphone-2023")
        assert acquired is True

    @pytest.mark.asyncio
    async def test_acquire_returns_false_when_locked(self):
        redis = _make_redis(set_return=None)    # SET NX returns None on collision
        acquired = await acquire_scrape_lock(redis, "apple-iphone-15-pro-smartphone-2023")
        assert acquired is False

    @pytest.mark.asyncio
    async def test_key_format(self):
        redis = _make_redis(set_return=b"OK")
        await acquire_scrape_lock(redis, "test-slug-x")
        redis.set.assert_awaited_once_with("fetch:test-slug-x", "locked", nx=True, ex=600)

    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        redis = _make_redis(set_return=b"OK")
        await acquire_scrape_lock(redis, "slug-a", ttl=300)
        _, kwargs = redis.set.call_args
        assert kwargs.get("ex") == 300 or redis.set.call_args[0][3] == 300 or True
        # Validate via the call args
        call_kwargs = redis.set.call_args.kwargs
        assert call_kwargs.get("ex") == 300 or redis.set.call_args[1].get("ex") == 300


class TestReleaseScrapeLock:
    @pytest.mark.asyncio
    async def test_release_deletes_key(self):
        redis = _make_redis()
        await release_scrape_lock(redis, "test-slug-x")
        redis.delete.assert_awaited_once_with("fetch:test-slug-x")


# ── Fetch status tests ────────────────────────────────────────────────────────


class TestProductFetchStatus:
    @pytest.mark.asyncio
    async def test_set_and_get_round_trip(self):
        redis = AsyncMock()
        redis.set.return_value = b"OK"
        redis.get.return_value = "fetching"

        await set_product_fetch_status(redis, "test-slug", "fetching")
        redis.set.assert_awaited_once()

        status = await get_product_fetch_status(redis, "test-slug")
        assert status == "fetching"

    @pytest.mark.asyncio
    async def test_key_format_uses_slug(self):
        redis = AsyncMock()
        redis.set.return_value = b"OK"
        await set_product_fetch_status(redis, "apple-iphone-15-smartphone-2023", "ready")
        call_args = redis.set.call_args
        assert call_args[0][0] == "fetch_status:apple-iphone-15-smartphone-2023"

    @pytest.mark.asyncio
    async def test_pre_scrape_returns_none(self):
        redis = _make_redis(get_return=None)
        status = await get_product_fetch_status(redis, "unknown-product-slug")
        assert status is None

    @pytest.mark.asyncio
    async def test_failed_status_set(self):
        redis = AsyncMock()
        redis.set.return_value = b"OK"
        await set_product_fetch_status(redis, "test-slug", "failed")
        call_args = redis.set.call_args
        assert call_args[0][1] == "failed"


# ── Embed lock tests ──────────────────────────────────────────────────────────


class TestAcquireEmbedLock:
    @pytest.mark.asyncio
    async def test_acquire_returns_true_when_free(self):
        redis = _make_redis(set_return=b"OK")
        acquired = await acquire_embed_lock(redis, product_id=42)
        assert acquired is True

    @pytest.mark.asyncio
    async def test_acquire_returns_false_when_locked(self):
        redis = _make_redis(set_return=None)
        acquired = await acquire_embed_lock(redis, product_id=42)
        assert acquired is False

    @pytest.mark.asyncio
    async def test_key_format(self):
        redis = _make_redis(set_return=b"OK")
        await acquire_embed_lock(redis, product_id=99)
        redis.set.assert_awaited_once_with("embed:99:specs", "locked", nx=True, ex=300)

    @pytest.mark.asyncio
    async def test_release_embed_lock(self):
        redis = _make_redis()
        await release_embed_lock(redis, product_id=99)
        redis.delete.assert_awaited_once_with("embed:99:specs")


# ── Idempotency: double-acquire ───────────────────────────────────────────────


class TestDoubleAcquire:
    @pytest.mark.asyncio
    async def test_double_scrape_lock_returns_false_second(self):
        """Simulates two concurrent workers: first gets True, second gets False."""
        call_count = 0

        async def mock_set(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return b"OK" if call_count == 1 else None  # second call returns None

        redis = AsyncMock()
        redis.set.side_effect = mock_set

        first = await acquire_scrape_lock(redis, "same-slug")
        second = await acquire_scrape_lock(redis, "same-slug")

        assert first is True
        assert second is False
