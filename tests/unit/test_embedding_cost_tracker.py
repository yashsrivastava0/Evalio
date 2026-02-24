"""Unit tests — embedding cost tracker.

Tests:
    - increment_embedding_metric calls Redis INCR
    - get_embedding_metrics computes skip_rate correctly
    - get_embedding_metrics computes avg_latency_ms correctly
    - Returns zero-safe defaults on Redis error
    - Reset deletes all counter keys
    - _ALL_EMB_METRICS_KEYS has correct prefix
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.embedding_cost_tracker import (
    increment_embedding_metric,
    get_embedding_metrics,
    reset_embedding_metrics,
    EMB_METRICS_TOTAL_GENERATED,
    EMB_METRICS_TOTAL_SKIPPED,
    EMB_METRICS_TOTAL_LATENCY_MS,
    EMB_METRICS_VERSION_RECOMPUTES,
    _ALL_EMB_METRICS_KEYS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.incr = AsyncMock()
    r.mget = AsyncMock(return_value=[None, None, None, None])
    r.delete = AsyncMock()
    return r


# ── increment_embedding_metric ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_increment_calls_redis_incr(mock_redis):
    """increment_embedding_metric must call Redis INCR."""
    await increment_embedding_metric(mock_redis, EMB_METRICS_TOTAL_GENERATED)
    mock_redis.incr.assert_called_once_with(EMB_METRICS_TOTAL_GENERATED, 1)


@pytest.mark.asyncio
async def test_increment_with_custom_amount(mock_redis):
    """increment_embedding_metric supports custom amount."""
    await increment_embedding_metric(mock_redis, EMB_METRICS_TOTAL_LATENCY_MS, 95)
    mock_redis.incr.assert_called_once_with(EMB_METRICS_TOTAL_LATENCY_MS, 95)


@pytest.mark.asyncio
async def test_increment_handles_redis_error(mock_redis):
    """increment_embedding_metric must not raise on Redis failure."""
    mock_redis.incr.side_effect = ConnectionError("Redis down")
    await increment_embedding_metric(mock_redis, EMB_METRICS_TOTAL_GENERATED)
    # No exception raised — swallowed silently


# ── get_embedding_metrics ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_all_zeros_when_empty(mock_redis):
    """All zeros when no data in Redis."""
    mock_redis.mget.return_value = [None, None, None, None]
    metrics = await get_embedding_metrics(mock_redis)

    assert metrics["total_generated"] == 0
    assert metrics["total_skipped"] == 0
    assert metrics["avg_latency_ms"] == 0.0
    assert metrics["skip_rate"] == 0.0
    assert metrics["version_recomputes"] == 0


@pytest.mark.asyncio
async def test_skip_rate_computation(mock_redis):
    """skip_rate = skipped / (generated + skipped)."""
    # generated=80, skipped=20, total_latency_ms=8000, version_recomputes=3
    mock_redis.mget.return_value = ["80", "20", "8000", "3"]
    metrics = await get_embedding_metrics(mock_redis)

    assert metrics["total_generated"] == 80
    assert metrics["total_skipped"] == 20
    assert metrics["skip_rate"] == pytest.approx(0.2, abs=0.001)  # 20/100


@pytest.mark.asyncio
async def test_avg_latency_computation(mock_redis):
    """avg_latency_ms = total_latency_ms / total_generated."""
    # generated=10, skipped=0, total_latency_ms=950
    mock_redis.mget.return_value = ["10", "0", "950", "0"]
    metrics = await get_embedding_metrics(mock_redis)

    assert metrics["avg_latency_ms"] == pytest.approx(95.0, abs=0.01)


@pytest.mark.asyncio
async def test_no_div_zero_when_no_generated(mock_redis):
    """avg_latency_ms should be 0.0 when generated=0 (no division by zero)."""
    mock_redis.mget.return_value = ["0", "10", "0", "0"]
    metrics = await get_embedding_metrics(mock_redis)
    assert metrics["avg_latency_ms"] == 0.0


@pytest.mark.asyncio
async def test_estimated_cost_is_positive(mock_redis):
    """Cost estimate should be positive when embeddings were generated."""
    mock_redis.mget.return_value = ["500", "100", "45000", "2"]
    metrics = await get_embedding_metrics(mock_redis)
    assert metrics["estimated_cost_usd"] > 0.0


@pytest.mark.asyncio
async def test_metrics_returns_zeros_on_redis_error(mock_redis):
    """get_embedding_metrics must return safe defaults on Redis error."""
    mock_redis.mget.side_effect = ConnectionError("Redis down")
    metrics = await get_embedding_metrics(mock_redis)

    assert metrics["total_generated"] == 0
    assert metrics["skip_rate"] == 0.0
    assert metrics["avg_latency_ms"] == 0.0


# ── reset_embedding_metrics ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_deletes_all_keys(mock_redis):
    """reset_embedding_metrics must delete all metric keys."""
    await reset_embedding_metrics(mock_redis)
    mock_redis.delete.assert_called_once_with(*_ALL_EMB_METRICS_KEYS)


@pytest.mark.asyncio
async def test_reset_handles_redis_error(mock_redis):
    """reset_embedding_metrics must not raise on Redis failure."""
    mock_redis.delete.side_effect = ConnectionError("Redis down")
    await reset_embedding_metrics(mock_redis)  # No exception


# ── Key format validation ──────────────────────────────────────────────────────


def test_all_emb_metrics_keys_have_correct_prefix():
    """All embedding metric keys must start with 'emb_metrics:'."""
    for key in _ALL_EMB_METRICS_KEYS:
        assert key.startswith("emb_metrics:"), f"Key '{key}' missing 'emb_metrics:' prefix"


def test_all_emb_metrics_keys_count():
    """Must have exactly 4 embedding metric keys."""
    assert len(_ALL_EMB_METRICS_KEYS) == 4


def test_all_keys_are_unique():
    """All metric keys must be unique."""
    assert len(_ALL_EMB_METRICS_KEYS) == len(set(_ALL_EMB_METRICS_KEYS))


# ── Raw data passthrough ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_raw_values_in_metrics(mock_redis):
    """metrics['raw'] must contain the raw counter values."""
    mock_redis.mget.return_value = ["50", "30", "4500", "5"]
    metrics = await get_embedding_metrics(mock_redis)

    raw = metrics["raw"]
    assert raw["total_generated"] == 50
    assert raw["total_skipped"] == 30
    assert raw["total_latency_ms"] == 4500
    assert raw["version_recomputes"] == 5
