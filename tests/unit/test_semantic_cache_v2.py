"""Unit tests — Semantic Cache v2 (Phase 6 key format).

Tests:
    - build_semantic_cache_key includes embedding_type prefix
    - "specs" and "review" keys are different for same hashes
    - Cache get/set round-trip with new key format
    - Exact hash match only (no cosine at cache layer)
    - Cache miss returns None
    - Freshness gate: TTL-stale cache returns None and deletes key
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import orjson
import pytest

from app.services.cache import (
    build_semantic_cache_key,
    get_semantic_cache,
    set_semantic_cache,
)


# ── Key format tests ──────────────────────────────────────────────────────────


def test_key_includes_embedding_type_specs():
    """Cache key must contain 'semantic:specs:v<N>:' prefix."""
    key = build_semantic_cache_key("ehash", "fhash", "specs")
    assert "semantic:specs:" in key
    assert ":specs:v" in key


def test_key_includes_embedding_type_review():
    """Cache key must contain 'semantic:review:v<N>:' prefix."""
    key = build_semantic_cache_key("ehash", "fhash", "review")
    assert "semantic:review:" in key
    assert ":review:v" in key


def test_specs_and_review_keys_differ():
    """Same embedding_hash + filter_hash must produce different keys by type."""
    specs_key = build_semantic_cache_key("ehash", "fhash", "specs")
    review_key = build_semantic_cache_key("ehash", "fhash", "review")
    assert specs_key != review_key


def test_key_is_exact_hash_not_cosine():
    """Key is SHA-256 derived — same inputs always produce same key."""
    k1 = build_semantic_cache_key("eh", "fh", "specs")
    k2 = build_semantic_cache_key("eh", "fh", "specs")
    assert k1 == k2


def test_default_embedding_type_is_specs():
    """Default embedding_type is 'specs' when not specified."""
    key_explicit = build_semantic_cache_key("eh", "fh", "specs")
    key_default = build_semantic_cache_key("eh", "fh")
    assert key_explicit == key_default


def test_key_structure():
    """Key must have format: semantic:<type>:v<version>:<24-char-hex> (4 parts)."""
    from app.core.constants import EMBEDDING_VERSION
    key = build_semantic_cache_key("embedding_hash_abc", "filter_hash_xyz", "specs")
    parts = key.split(":")
    assert len(parts) == 4, f"Expected 4 parts, got {len(parts)}: {key}"
    assert parts[0] == "semantic"
    assert parts[1] == "specs"
    assert parts[2] == f"v{EMBEDDING_VERSION}"
    assert len(parts[3]) == 24
    assert all(c in "0123456789abcdef" for c in parts[3])


def test_embedding_version_baked_into_key():
    """Fix-8: Different EMBEDDING_VERSION values must produce different keys.

    This ensures a version bump naturally expires all semantic cache entries
    without requiring manual Redis FLUSHDB.
    """
    from app.core.constants import EMBEDDING_VERSION
    key_v1 = build_semantic_cache_key("eh", "fh", "specs")
    # Verify version appears in key — if bumped, key changes
    assert f"v{EMBEDDING_VERSION}" in key_v1
    # Simulate what a v2 key would look like
    import hashlib
    combined_v2 = f"eh:fh:v{EMBEDDING_VERSION + 1}"
    digest_v2 = hashlib.sha256(combined_v2.encode()).hexdigest()[:24]
    key_v2 = f"semantic:specs:v{EMBEDDING_VERSION + 1}:{digest_v2}"
    assert key_v1 != key_v2


# ── Cache get/set round-trip ──────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    return AsyncMock()


@pytest.mark.asyncio
async def test_cache_miss_returns_none(mock_redis):
    """get_semantic_cache returns None when key not in Redis."""
    mock_redis.get.return_value = None
    result = await get_semantic_cache(mock_redis, "eh", "fh", embedding_type="specs")
    assert result is None


@pytest.mark.asyncio
async def test_set_then_get_round_trip(mock_redis):
    """set_semantic_cache stores, get_semantic_cache retrieves."""
    stored = {}

    async def fake_set(key, value, ex=None):
        stored[key] = value

    async def fake_get(key):
        return stored.get(key)

    mock_redis.set.side_effect = fake_set
    mock_redis.get.side_effect = fake_get
    mock_redis.incr = AsyncMock()

    payload = {"ranked": [1, 2, 3], "ranked_product_ids": [1, 2, 3]}
    await set_semantic_cache(mock_redis, "eh", "fh", payload, embedding_type="specs")
    result = await get_semantic_cache(mock_redis, "eh", "fh", embedding_type="specs")

    assert result is not None
    assert result["ranked"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_specs_and_review_caches_are_isolated(mock_redis):
    """Specs and review caches must not interfere with each other."""
    stored = {}

    async def fake_set(key, value, ex=None):
        stored[key] = value

    async def fake_get(key):
        return stored.get(key)

    mock_redis.set.side_effect = fake_set
    mock_redis.get.side_effect = fake_get
    mock_redis.incr = AsyncMock()

    specs_payload = {"source": "specs", "ranked_product_ids": [1]}
    review_payload = {"source": "review", "ranked_product_ids": [2]}

    await set_semantic_cache(mock_redis, "eh", "fh", specs_payload, embedding_type="specs")
    await set_semantic_cache(mock_redis, "eh", "fh", review_payload, embedding_type="review")

    specs_result = await get_semantic_cache(mock_redis, "eh", "fh", embedding_type="specs")
    review_result = await get_semantic_cache(mock_redis, "eh", "fh", embedding_type="review")

    assert specs_result["source"] == "specs"
    assert review_result["source"] == "review"


# ── Freshness gate: TTL-based staleness ───────────────────────────────────────


@pytest.mark.asyncio
async def test_stale_cache_returns_none_and_deletes(mock_redis):
    """Stale cache entry (age > max_age_hours) must return None and delete key."""
    # Simulate a cache entry 48 hours old (stale)
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    stale_data = orjson.dumps({"ranked": [1, 2], "cached_at": stale_time, "ranked_product_ids": [1, 2]})
    mock_redis.get.return_value = stale_data
    mock_redis.delete = AsyncMock()

    result = await get_semantic_cache(
        mock_redis, "eh", "fh",
        embedding_type="specs",
        max_age_hours=1,  # expire after 1 hour
    )

    assert result is None
    mock_redis.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_fresh_cache_returns_data(mock_redis):
    """Fresh cache entry (age < max_age_hours) must be returned."""
    fresh_time = datetime.now(timezone.utc).isoformat()
    fresh_data = orjson.dumps({
        "ranked": [1, 2],
        "cached_at": fresh_time,
        "ranked_product_ids": [1, 2],
    })
    mock_redis.get.return_value = fresh_data
    mock_redis.incr = AsyncMock()

    result = await get_semantic_cache(
        mock_redis, "eh", "fh",
        embedding_type="specs",
        max_age_hours=24,
    )

    assert result is not None
    assert result["ranked"] == [1, 2]


@pytest.mark.asyncio
async def test_cache_handles_corrupt_data_gracefully(mock_redis):
    """Corrupt cache entry must not crash — return None."""
    mock_redis.get.return_value = b"not valid json!!"
    result = await get_semantic_cache(mock_redis, "eh", "fh")
    assert result is None
