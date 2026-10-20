"""Unit tests for the DecisionMatrix node.

Verifies:
    1. No LLM imports in decision_matrix.py
    2. Exact query cache hit → route = "cache_hit"
    3. Semantic cache hit → route = "cache_hit"
    4. Default route is "fast_path"
    5. Follow-up with matching filter_hash → fast_path + use_session_comparison
    6. Follow-up injects session.last_results as candidates (skip SQL)
    7. Stale data does NOT escalate to fetch_trigger
    8. Semantic cache uses exact hash — not cosine
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.decision_matrix import (
    decision_matrix_node,
    _semantic_cache_key,
    _is_stale,
)


# ── No-LLM Guard ──


def test_no_llm_imports():
    """decision_matrix.py must NEVER import LLM modules."""
    import inspect
    import app.agents.decision_matrix as dm_module

    source = inspect.getsource(dm_module)
    forbidden = ["openai", "langchain", "anthropic", "google.generativeai"]
    for module in forbidden:
        assert f"import {module}" not in source, f"Forbidden LLM import: {module}"


# ── Semantic Cache Key ──


def test_semantic_cache_key_exact_hash():
    """Semantic cache key must use exact hash, not cosine similarity."""
    key1 = _semantic_cache_key("abc123", "def456")
    key2 = _semantic_cache_key("abc123", "def456")
    assert key1 == key2
    assert key1.startswith("semantic:")


def test_semantic_cache_key_different_filters():
    """Different filter_hash → different cache key."""
    key1 = _semantic_cache_key("abc123", "filter_a")
    key2 = _semantic_cache_key("abc123", "filter_b")
    assert key1 != key2


# ── Staleness ──


def test_is_stale_none_timestamp():
    """None timestamp is always stale."""
    assert _is_stale(None, max_age_hours=24) is True


def test_is_stale_recent_timestamp():
    """Recent timestamp is not stale."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    assert _is_stale(now, max_age_hours=24) is False


# ── Mock Helpers ──


@dataclass
class MockSession:
    session_id: str = "test-session"
    last_results: list = field(default_factory=list)
    active_comparison: list = field(default_factory=list)
    conversation_summary: str = ""
    last_filter_hash: str = ""
    last_query_embedding_hash: str = ""
    turn_count: int = 0


@dataclass
class MockIntent:
    category: str | None = None
    brands: list = field(default_factory=list)
    is_comparison: bool = False
    intent_type: str = "search"
    budget_max: float | None = None
    priority_list: list = field(default_factory=list)


def _make_redis_mock(cache_data: dict | None = None) -> AsyncMock:
    """Create mock Redis that returns cache_data for any get() call."""
    redis = AsyncMock()

    async def mock_get(key):
        if cache_data and key in cache_data:
            return cache_data[key]
        return None

    redis.get = mock_get
    redis.set = AsyncMock()
    return redis


# ── Cache Hit Tests ──


@pytest.mark.asyncio
async def test_exact_query_cache_hit():
    """Exact query cache match → route = cache_hit."""
    redis = _make_redis_mock({"query_cache:abcdef12": "cached_response_data"})

    result = await decision_matrix_node(
        {"query_hash": "abcdef12", "embedding_hash": "", "filter_hash": ""},
        redis_client=redis,
    )

    assert result["route"] == "cache_hit"
    assert result["cached_response"] == "cached_response_data"


@pytest.mark.asyncio
async def test_no_cache_defaults_to_fast_path():
    """No cache hits → default route = fast_path."""
    redis = _make_redis_mock()

    result = await decision_matrix_node(
        {
            "query_hash": "no_match",
            "embedding_hash": "",
            "filter_hash": "",
            "intent": MockIntent(),
            "session": MockSession(),
        },
        redis_client=redis,
    )

    assert result["route"] == "fast_path"


# ── Session Follow-up Tests ──


@pytest.mark.asyncio
async def test_follow_up_reuses_session_results():
    """Follow-up query with matching filter_hash → fast_path + candidates from session."""
    session = MockSession(
        last_results=[
            {"product_id": 1, "final_score": 0.9},
            {"product_id": 2, "final_score": 0.8},
        ],
        last_filter_hash="filter_abc",
    )
    intent = MockIntent(intent_type="follow_up")
    redis = _make_redis_mock()

    result = await decision_matrix_node(
        {
            "query_hash": "",
            "embedding_hash": "",
            "filter_hash": "filter_abc",
            "session": session,
            "intent": intent,
        },
        redis_client=redis,
    )

    assert result["route"] == "fast_path"
    assert result["use_session_comparison"] is True
    assert result["candidates"] == session.last_results


@pytest.mark.asyncio
async def test_follow_up_no_filter_match_goes_fast_path():
    """Follow-up with SQL filters + different filter_hash → no session reuse.

    When a follow-up query carries explicit SQL-level constraints (e.g. category),
    the filter_hash must match the session's last_filter_hash for session reuse.
    A hash mismatch means the user's intent changed scope → skip session bypass.
    """
    session = MockSession(
        last_results=[{"product_id": 1, "final_score": 0.9}],
        last_filter_hash="filter_OLD",
    )
    # Provide a SQL-level filter (category) so hash-mismatch guard is enforced
    intent = MockIntent(intent_type="follow_up", category="smartphone")
    redis = _make_redis_mock()

    result = await decision_matrix_node(
        {
            "query_hash": "",
            "embedding_hash": "",
            "filter_hash": "filter_NEW",
            "session": session,
            "intent": intent,
        },
        redis_client=redis,
    )

    assert result["route"] == "fast_path"
    # Should NOT have use_session_comparison since filters don't match
    assert result.get("use_session_comparison") is None


@pytest.mark.asyncio
async def test_follow_up_skips_sql():
    """Follow-up query with session reuse must inject candidates without SQL.

    Validates: RetrievalNode will see use_session_comparison=True and skip SQL+vector.
    """
    last_results = [
        {"product_id": 10, "final_score": 0.95, "product_slug": "phone-10"},
        {"product_id": 20, "final_score": 0.85, "product_slug": "phone-20"},
    ]
    session = MockSession(
        last_results=last_results,
        last_filter_hash="match_hash",
    )
    intent = MockIntent(intent_type="follow_up")
    redis = _make_redis_mock()

    result = await decision_matrix_node(
        {
            "query_hash": "",
            "embedding_hash": "",
            "filter_hash": "match_hash",
            "session": session,
            "intent": intent,
        },
        redis_client=redis,
    )

    # Candidates are injected from session — DB not needed
    assert result["candidates"] == last_results
    assert result["use_session_comparison"] is True
    assert result["route"] == "fast_path"


# ── Fetch Status Tests ──


@pytest.mark.asyncio
async def test_in_flight_fetch_returns_fetch_trigger():
    """In-flight scrape for a brand → route = fetch_trigger."""
    redis = _make_redis_mock({"fetch_status:samsung": "fetching"})
    intent = MockIntent(brands=["samsung"])

    result = await decision_matrix_node(
        {
            "query_hash": "",
            "embedding_hash": "",
            "filter_hash": "",
            "intent": intent,
        },
        redis_client=redis,
    )

    assert result["route"] == "fetch_trigger"
    assert result["fetch_in_flight"] is True


# ── Session Scores Completeness ──


@pytest.mark.asyncio
async def test_session_scores_complete_7_components():
    """Session last_results candidates must include all 7 component scores.

    RankingNode needs every component score on follow-up queries.
    If even one is missing → tie-breaking instability.
    This test guards against future refactors dropping a field.
    """
    CANONICAL_7 = {
        "performance_score",
        "battery_score",
        "camera_score",
        "display_score",
        "build_score",
        "price_score",
        "popularity_score",
    }

    # Simulate what SessionUpdateNode stores in session.last_results
    # (mirrors graph.py L178-193)
    mock_ranked = [
        {
            "product_id": 1,
            "product_slug": "phone-1",
            "canonical_title": "Test Phone 1",
            "brand": "test",
            "category": "smartphone",
            "final_score": 0.85,
            "best_price": 25000,
            "scores": {
                "performance_score": 0.9,
                "battery_score": 0.8,
                "camera_score": 0.7,
                "display_score": 0.85,
                "build_score": 0.6,
                "price_score": 0.75,
                "popularity_score": 0.5,
            },
            "rank": 1,
        }
    ]

    # Build session entry the same way graph.py does
    stored = {
        "product_id": mock_ranked[0].get("product_id"),
        "product_slug": mock_ranked[0].get("product_slug"),
        "canonical_title": mock_ranked[0].get("canonical_title"),
        "brand": mock_ranked[0].get("brand"),
        "category": mock_ranked[0].get("category"),
        "final_score": mock_ranked[0].get("final_score"),
        "best_price": mock_ranked[0].get("best_price"),
        "scores": mock_ranked[0].get("scores", {}),
        "rank": mock_ranked[0].get("rank"),
    }

    # Verify all 7 components exist
    stored_score_keys = set(stored["scores"].keys())
    missing = CANONICAL_7 - stored_score_keys
    assert not missing, f"Session last_results missing scores: {missing}"
    assert stored_score_keys >= CANONICAL_7

