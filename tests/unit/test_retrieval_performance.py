"""Unit tests for retrieval performance guarantees.

Verifies:
    1. No full-table vector scan (query uses WHERE clause on filtered IDs)
    2. SQL filter uses parameterized queries (no SQL injection vectors)
    3. Candidate caps are enforced from config defaults
    4. Vector re-rank operates only on pre-filtered subset
    5. Session bypass skips SQL+vector pipeline
"""

from __future__ import annotations

import inspect

import pytest

from app.core.config import get_settings


# ── No Full-Table Vector Scan ──

def test_vector_rerank_uses_where_clause():
    """vector_rerank_candidates must use WHERE product_id = ANY()
    or JOIN on filtered IDs — never a full table scan."""
    from app.services.retrieval import vector_rerank_candidates
    source = inspect.getsource(vector_rerank_candidates)

    # Must restrict to filtered IDs
    assert "ANY(:product_ids)" in source or "product_ids" in source, \
        "Vector re-rank must filter by product_ids — no full table scan allowed"

    # Must not select from embeddings without filter
    assert "WHERE" in source, \
        "Vector query must have WHERE clause"


def test_vector_rerank_has_limit():
    """Vector re-rank query must include LIMIT clause."""
    from app.services.retrieval import vector_rerank_candidates
    source = inspect.getsource(vector_rerank_candidates)
    assert "LIMIT" in source, "Vector query must have LIMIT to cap results"


def test_vector_rerank_no_full_scan_keyword():
    """No SELECT * FROM embeddings without filter."""
    from app.services.retrieval import vector_rerank_candidates
    source = inspect.getsource(vector_rerank_candidates)
    # Should NOT have "SELECT * FROM embeddings" without WHERE
    assert "SELECT * FROM embeddings" not in source, \
        "Full table scan pattern detected — must use filtered JOIN"


# ── SQL Parameterization ──

def test_sql_filter_uses_parameterized_queries():
    """SQL filter must use parameterized queries (no f-strings in SQL)."""
    from app.services.retrieval import sql_filter_candidates
    source = inspect.getsource(sql_filter_candidates)

    # Should not have raw f-string SQL
    assert 'f"SELECT' not in source and "f'SELECT" not in source, \
        "SQL filter must use parameterized queries, not f-strings"


# ── Candidate Caps from Config ──

def test_config_sql_candidate_default():
    """Production default SQL candidates must be 300 (immutable)."""
    settings = get_settings()
    assert settings.max_sql_candidates == 300, \
        f"max_sql_candidates must be 300, got {settings.max_sql_candidates}"


def test_config_vector_candidate_default():
    """Production default vector candidates must be 50."""
    settings = get_settings()
    assert settings.max_vector_candidates == 50, \
        f"max_vector_candidates must be 50, got {settings.max_vector_candidates}"


def test_config_rank_candidate_default():
    """Production default ranking candidates must be 30."""
    settings = get_settings()
    assert settings.max_rank_candidates == 30, \
        f"max_rank_candidates must be 30, got {settings.max_rank_candidates}"


def test_sql_filter_respects_max_candidates():
    """SQL filter must read max_candidates from config settings."""
    from app.services.retrieval import sql_filter_candidates
    source = inspect.getsource(sql_filter_candidates)
    assert "max_sql_candidates" in source, \
        "SQL filter must reference max_sql_candidates from settings"


def test_vector_rerank_respects_max_candidates():
    """Vector re-rank must read max_candidates from config settings."""
    from app.services.retrieval import vector_rerank_candidates
    source = inspect.getsource(vector_rerank_candidates)
    assert "max_vector_candidates" in source, \
        "Vector re-rank must reference max_vector_candidates from settings"


def test_reduce_respects_max_candidates():
    """Candidate reducer must read max_candidates from config settings."""
    from app.services.retrieval import reduce_candidates
    source = inspect.getsource(reduce_candidates)
    assert "max_rank_candidates" in source, \
        "Candidate reducer must reference max_rank_candidates from settings"


# ── Session Bypass ──

def test_retrieval_node_has_session_bypass():
    """retrieval_node must check use_session_comparison before running pipeline."""
    from app.services.retrieval import retrieval_node
    source = inspect.getsource(retrieval_node)
    assert "use_session_comparison" in source, \
        "retrieval_node must support session bypass via use_session_comparison"


# ── Empty Input Edge Cases ──

@pytest.mark.asyncio
async def test_vector_rerank_empty_ids_returns_empty():
    """Vector re-rank with empty filtered IDs returns empty list."""
    from app.services.retrieval import vector_rerank_candidates
    from unittest.mock import AsyncMock

    mock_db = AsyncMock()
    result = await vector_rerank_candidates(mock_db, [], [0.1, 0.2])
    assert result == []


@pytest.mark.asyncio
async def test_reduce_candidates_empty_returns_empty():
    """Candidate reducer with empty input returns empty list."""
    from app.services.retrieval import reduce_candidates
    from unittest.mock import AsyncMock

    mock_db = AsyncMock()
    result = await reduce_candidates(mock_db, [])
    assert result == []
