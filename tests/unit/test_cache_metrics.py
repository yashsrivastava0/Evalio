"""Unit tests for cache metrics instrumentation.

Verifies:
    1. Redis counters increment correctly on cache operations
    2. get_cache_metrics returns correct ratios
    3. reset_cache_metrics zeroes all counters
    4. Metrics are process-safe (Redis-based, not in-memory)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.cache import (
    increment_cache_metric,
    get_cache_metrics,
    reset_cache_metrics,
    METRICS_EXACT_QUERY_HITS,
    METRICS_SEMANTIC_HITS,
    METRICS_DB_FAST_PATH_HITS,
    METRICS_FETCH_TRIGGER_COUNT,
    METRICS_TOTAL_REQUESTS,
    _ALL_METRICS_KEYS,
)


# ── Fixtures ──

@pytest.fixture
def mock_redis():
    """Create a mock Redis client with async methods."""
    redis = AsyncMock()
    redis.incr = AsyncMock()
    redis.mget = AsyncMock(return_value=[None, None, None, None, None])
    redis.delete = AsyncMock()
    return redis


# ── Increment Tests ──

@pytest.mark.asyncio
async def test_increment_calls_redis_incr(mock_redis):
    """increment_cache_metric should call Redis INCR."""
    await increment_cache_metric(mock_redis, METRICS_SEMANTIC_HITS)
    mock_redis.incr.assert_called_once_with(METRICS_SEMANTIC_HITS, 1)


@pytest.mark.asyncio
async def test_increment_with_custom_amount(mock_redis):
    """increment_cache_metric with amount > 1."""
    await increment_cache_metric(mock_redis, METRICS_TOTAL_REQUESTS, 5)
    mock_redis.incr.assert_called_once_with(METRICS_TOTAL_REQUESTS, 5)


@pytest.mark.asyncio
async def test_increment_handles_redis_error(mock_redis):
    """increment_cache_metric should not raise on Redis errors."""
    mock_redis.incr.side_effect = ConnectionError("Redis down")
    # Should not raise
    await increment_cache_metric(mock_redis, METRICS_SEMANTIC_HITS)


# ── Get Metrics Tests ──

@pytest.mark.asyncio
async def test_get_metrics_empty(mock_redis):
    """With no data, all ratios should be 0.0."""
    mock_redis.mget.return_value = [None, None, None, None, None]
    metrics = await get_cache_metrics(mock_redis)

    assert metrics["exact_hit_ratio"] == 0.0
    assert metrics["semantic_hit_ratio"] == 0.0
    assert metrics["fast_path_ratio"] == 0.0
    assert metrics["fetch_ratio"] == 0.0
    assert metrics["raw"]["total_requests"] == 0


@pytest.mark.asyncio
async def test_get_metrics_with_data(mock_redis):
    """Ratios should compute correctly from raw counters."""
    # exact=10, semantic=5, fast_path=30, fetch=5, total=50
    mock_redis.mget.return_value = ["10", "5", "30", "5", "50"]
    metrics = await get_cache_metrics(mock_redis)

    assert metrics["exact_hit_ratio"] == 0.2     # 10/50
    assert metrics["semantic_hit_ratio"] == 0.1   # 5/50
    assert metrics["fast_path_ratio"] == 0.6      # 30/50
    assert metrics["fetch_ratio"] == 0.1          # 5/50
    assert metrics["raw"]["total_requests"] == 50
    assert metrics["raw"]["exact_query_hits"] == 10
    assert metrics["raw"]["semantic_hits"] == 5


@pytest.mark.asyncio
async def test_get_metrics_handles_redis_error(mock_redis):
    """get_cache_metrics should return zeros on Redis error."""
    mock_redis.mget.side_effect = ConnectionError("Redis down")
    metrics = await get_cache_metrics(mock_redis)

    assert metrics["exact_hit_ratio"] == 0.0
    assert metrics["raw"]["total_requests"] == 0


# ── Reset Tests ──

@pytest.mark.asyncio
async def test_reset_deletes_all_keys(mock_redis):
    """reset_cache_metrics should delete all metrics keys."""
    await reset_cache_metrics(mock_redis)
    mock_redis.delete.assert_called_once_with(*_ALL_METRICS_KEYS)


@pytest.mark.asyncio
async def test_reset_handles_redis_error(mock_redis):
    """reset_cache_metrics should not raise on Redis errors."""
    mock_redis.delete.side_effect = ConnectionError("Redis down")
    # Should not raise
    await reset_cache_metrics(mock_redis)


# ── Metrics Keys Validation ──

def test_all_metrics_keys_count():
    """Must have exactly 5 metrics keys."""
    assert len(_ALL_METRICS_KEYS) == 5


def test_metrics_keys_have_prefix():
    """All metrics keys must start with 'metrics:' prefix."""
    for key in _ALL_METRICS_KEYS:
        assert key.startswith("metrics:"), f"Key '{key}' missing 'metrics:' prefix"
