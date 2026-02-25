"""Phase 7 — Multi-product comparison limits unit tests.

Verifies:
    - Reject > 4 products (ComparisonIntegrityError)
    - Reject duplicate IDs (ComparisonIntegrityError)
    - 2, 3, 4 product comparisons produce valid Schema v2
    - compare_ids subset preserves ranking order
    - Session validation rejects drifted filter_hash
    - Session validation rejects IDs not in last_results
"""

from __future__ import annotations

import pytest

from app.agents.comparison_guard import (
    ComparisonIntegrityError,
    build_comparison_table,
    comparison_guard_node,
    CANONICAL_METRICS,
)
from app.agents.decision_matrix import _validate_session_compare_ids
from app.agents.session_context import SessionData


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_product(product_id: int, final_score: float = 0.75) -> dict:
    scores = {
        "performance_score": 0.7,
        "battery_score": 0.6,
        "camera_score": 0.8,
        "display_score": 0.7,
        "build_score": 0.5,
        "price_score": 0.6,
        "popularity_score": 0.4,
    }
    return {
        "product_id": product_id,
        "product_slug": f"product-{product_id}",
        "canonical_title": f"Product {product_id}",
        "final_score": final_score,
        "rank": product_id,
        "scores": scores,
    }


def _make_session(last_result_ids: list[int], filter_hash: str = "abc123") -> SessionData:
    session = SessionData(session_id="test-session")
    session.last_results = [{"product_id": pid} for pid in last_result_ids]
    session.last_filter_hash = filter_hash
    return session


# ── Multi-Product Count Limits ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_more_than_4_products():
    """compare_ids with 5+ products must raise ComparisonIntegrityError."""
    ranked = [_make_product(i, 0.9 - i * 0.1) for i in range(1, 6)]
    state = {"ranked": ranked, "compare_ids": [1, 2, 3, 4, 5]}
    with pytest.raises(ComparisonIntegrityError, match="exceeds max 4"):
        await comparison_guard_node(state)


@pytest.mark.asyncio
async def test_four_product_comparison_works():
    """4 products is the maximum allowed — must succeed."""
    ranked = [_make_product(i, 0.9 - i * 0.1) for i in range(1, 5)]
    state = {"ranked": ranked, "compare_ids": [1, 2, 3, 4]}
    result = await comparison_guard_node(state)
    assert len(result["comparison"]["comparison_table"]) == 4


@pytest.mark.asyncio
async def test_three_product_comparison_works():
    """3 products must produce valid Schema v2."""
    ranked = [_make_product(i, 0.9 - i * 0.1) for i in range(1, 4)]
    state = {"ranked": ranked}
    result = await comparison_guard_node(state)
    assert len(result["comparison"]["comparison_table"]) == 3
    for m in CANONICAL_METRICS:
        assert m in result["comparison"]["diff_matrix"]


@pytest.mark.asyncio
async def test_two_product_comparison_works():
    """2 products must produce valid Schema v2 with all 7 metrics in diff_matrix."""
    ranked = [_make_product(1, 0.85), _make_product(2, 0.65)]
    state = {"ranked": ranked}
    result = await comparison_guard_node(state)
    assert len(result["comparison"]["comparison_table"]) == 2
    assert len(result["comparison"]["diff_matrix"]) == 7


@pytest.mark.asyncio
async def test_single_product_comparison_works():
    """1 product must produce valid Schema v2 with zero diffs."""
    ranked = [_make_product(1, 0.85)]
    state = {"ranked": ranked}
    result = await comparison_guard_node(state)
    assert len(result["comparison"]["comparison_table"]) == 1
    for metric, data in result["comparison"]["diff_matrix"].items():
        assert data["absolute_diff"] == 0.0
        assert data["winner"] is None


# ── Duplicate IDs ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_duplicate_ids():
    """Duplicate IDs in compare_ids must raise ComparisonIntegrityError."""
    ranked = [_make_product(1, 0.9), _make_product(2, 0.7)]
    state = {"ranked": ranked, "compare_ids": [1, 1, 2]}
    with pytest.raises(ComparisonIntegrityError, match="duplicates"):
        await comparison_guard_node(state)


@pytest.mark.asyncio
async def test_reject_all_duplicates():
    """All-same IDs in compare_ids must raise ComparisonIntegrityError."""
    ranked = [_make_product(1, 0.9)]
    state = {"ranked": ranked, "compare_ids": [1, 1, 1]}
    with pytest.raises(ComparisonIntegrityError):
        await comparison_guard_node(state)


# ── Out-of-Ranked IDs ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_out_of_ranked_id():
    """ID not present in ranked list must raise ComparisonIntegrityError."""
    ranked = [_make_product(1, 0.9), _make_product(2, 0.7)]
    state = {"ranked": ranked, "compare_ids": [1, 99]}
    with pytest.raises(ComparisonIntegrityError, match="not present in ranked"):
        await comparison_guard_node(state)


@pytest.mark.asyncio
async def test_all_compare_ids_must_be_in_ranked():
    """All supplied compare_ids must be present in ranked — partial match is not enough."""
    ranked = [_make_product(1, 0.9)]
    state = {"ranked": ranked, "compare_ids": [1, 2, 3]}  # 2, 3 not in ranked
    with pytest.raises(ComparisonIntegrityError):
        await comparison_guard_node(state)


# ── Subset Order Preservation ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_ids_preserves_ranking_order():
    """compare_ids must respect ranking order, NOT user-supplied ID order."""
    ranked = [
        _make_product(5, 0.90),
        _make_product(2, 0.80),
        _make_product(9, 0.70),
    ]
    # User specifies IDs in reverse rank order
    state = {"ranked": ranked, "compare_ids": [9, 5]}
    result = await comparison_guard_node(state)
    output_ids = [e["product_id"] for e in result["comparison"]["comparison_table"]]
    assert output_ids == [5, 9], (
        f"Ranking order violated: got {output_ids}, expected [5, 9]"
    )


@pytest.mark.asyncio
async def test_no_compare_ids_uses_full_ranked():
    """When compare_ids is absent, all ranked products are compared."""
    ranked = [_make_product(i, 0.9 - i * 0.1) for i in range(1, 4)]
    state = {"ranked": ranked}  # no compare_ids key
    result = await comparison_guard_node(state)
    assert len(result["comparison"]["comparison_table"]) == 3


# ── Session Validation (DecisionMatrixNode layer) ─────────────────────────────


def test_session_validation_ok():
    """Valid compare_ids in session with matching filter_hash must pass."""
    session = _make_session([1, 2, 3], filter_hash="hash1")
    is_valid, reason = _validate_session_compare_ids([1, 2], session, "hash1")
    assert is_valid is True
    assert reason == "ok"


def test_session_validation_filter_drift_rejected():
    """Filter hash drift must invalidate session comparison."""
    session = _make_session([1, 2, 3], filter_hash="old_hash")
    is_valid, reason = _validate_session_compare_ids([1, 2], session, "new_hash")
    assert is_valid is False
    assert reason == "filter_drift"


def test_session_validation_id_not_in_results():
    """ID not in session.last_results must be rejected."""
    session = _make_session([1, 2, 3], filter_hash="hash1")
    is_valid, reason = _validate_session_compare_ids([1, 99], session, "hash1")
    assert is_valid is False
    assert "99" in reason


def test_session_validation_no_session():
    """Missing session must return invalid."""
    is_valid, reason = _validate_session_compare_ids([1, 2], None, "hash1")
    assert is_valid is False
    assert reason == "no_session"


def test_session_validation_empty_filter_hash_skips_drift_check():
    """Empty filter_hash in session skips drift check (first-turn safety)."""
    session = _make_session([1, 2, 3], filter_hash="")  # no previous hash
    is_valid, reason = _validate_session_compare_ids([1, 2], session, "any_hash")
    assert is_valid is True


# ── Schema v2 Completeness ─────────────────────────────────────────────────────


def test_schema_v2_all_canonical_metrics():
    """build_comparison_table output must contain all 7 canonical metrics in diff_matrix."""
    products = [_make_product(1, 0.8), _make_product(2, 0.6)]
    result = build_comparison_table(products)
    assert set(result["diff_matrix"].keys()) == set(CANONICAL_METRICS)


def test_schema_v2_metric_wins_sum():
    """Sum of metric_wins across all products must equal number of non-tie metrics."""
    products = [_make_product(1, 0.8), _make_product(2, 0.6)]
    result = build_comparison_table(products)
    total_wins = sum(e["metric_wins"] for e in result["comparison_table"])
    non_tie_metrics = sum(
        1 for d in result["diff_matrix"].values() if d["winner"] is not None
    )
    assert total_wins == non_tie_metrics
