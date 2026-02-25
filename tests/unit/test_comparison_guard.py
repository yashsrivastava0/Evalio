"""Unit tests for the ComparisonGuard node — updated for Phase 7 Schema v2.

Tests:
    1. No LLM imports in comparison_guard.py
    2. validate_ranking_integrity raises on mismatch
    3. build_comparison_table returns Schema v2 structure
    4. Numeric diffs correct (absolute_diff, percent_diff, winner)
    5. comparison_guard_node returns GraphState-compatible dict
"""

from __future__ import annotations

import pytest

from app.agents.comparison_guard import (
    validate_ranking_integrity,
    build_comparison_table,
    comparison_guard_node,
    ComparisonGuardError,
    ComparisonIntegrityError,
    CANONICAL_METRICS,
)


# ── No-LLM Guard ──


def test_no_llm_imports():
    """comparison_guard.py must NEVER import LLM modules."""
    import inspect
    import app.agents.comparison_guard as cg_module

    source = inspect.getsource(cg_module)
    forbidden = ["openai", "langchain", "anthropic", "google.generativeai"]
    for module in forbidden:
        assert f"import {module}" not in source, f"Forbidden LLM import: {module}"


# ── Integrity Validation ──


def test_matching_ids_pass():
    """Identical ranked_ids should pass validation."""
    assert validate_ranking_integrity([1, 2, 3], [1, 2, 3]) is True


def test_mismatched_ids_raise():
    """Different ranked_ids must raise ComparisonGuardError."""
    with pytest.raises(ComparisonGuardError):
        validate_ranking_integrity([1, 2, 3], [1, 3, 2])


def test_empty_ids_pass():
    """Empty lists are valid (no candidates)."""
    assert validate_ranking_integrity([], []) is True


# ── Schema v2 Structure ──


def _make_ranked_product(product_id: int, final_score: float = 0.65, **score_overrides) -> dict:
    """Helper to create a ranked product with scores."""
    scores = {
        "performance_score": 0.7,
        "battery_score": 0.6,
        "camera_score": 0.8,
        "display_score": 0.7,
        "build_score": 0.5,
        "price_score": 0.6,
        "popularity_score": 0.4,
    }
    scores.update(score_overrides)
    return {
        "product_id": product_id,
        "product_slug": f"phone-{chr(ord('a') + product_id - 1)}",
        "canonical_title": f"Product {product_id}",
        "final_score": final_score,
        "rank": product_id,
        "scores": scores,
    }


def test_comparison_table_v2_keys():
    """build_comparison_table must return Schema v2 with comparison_table + diff_matrix."""
    ranked = [_make_ranked_product(1), _make_ranked_product(2)]
    result = build_comparison_table(ranked)

    assert "comparison_table" in result
    assert "diff_matrix" in result


def test_comparison_table_all_seven_metrics_in_diff_matrix():
    """diff_matrix must include all 7 canonical metrics."""
    ranked = [_make_ranked_product(1), _make_ranked_product(2)]
    result = build_comparison_table(ranked)

    for metric in CANONICAL_METRICS:
        assert metric in result["diff_matrix"], f"Missing metric in diff_matrix: {metric}"


def test_comparison_table_entry_structure():
    """Each comparison_table entry must have product_id, component_scores, metric_wins, total_score."""
    ranked = [_make_ranked_product(1, 0.8), _make_ranked_product(2, 0.6)]
    result = build_comparison_table(ranked)

    for entry in result["comparison_table"]:
        assert "product_id" in entry
        assert "component_scores" in entry
        assert "metric_wins" in entry
        assert "total_score" in entry


def test_comparison_table_numeric_diffs():
    """absolute_diff for performance_score must be correct."""
    ranked = [
        _make_ranked_product(1, 0.8, performance_score=0.9),
        _make_ranked_product(2, 0.6, performance_score=0.6),
    ]
    result = build_comparison_table(ranked)
    diff = result["diff_matrix"]["performance_score"]

    assert abs(diff["absolute_diff"] - 0.3) < 0.001
    assert diff["winner"] == "1"  # product_id as str


def test_comparison_table_winner_determination():
    """Winner determined by highest value per metric (as product_id str)."""
    ranked = [
        _make_ranked_product(1, 0.8, battery_score=0.4),
        _make_ranked_product(2, 0.6, battery_score=0.9),
    ]
    result = build_comparison_table(ranked)

    assert result["diff_matrix"]["battery_score"]["winner"] == "2"


def test_comparison_table_total_score_not_recomputed():
    """total_score in comparison_table must equal product['final_score'] exactly."""
    ranked = [_make_ranked_product(1, 0.87654321)]
    result = build_comparison_table(ranked)
    assert abs(result["comparison_table"][0]["total_score"] - 0.87654321) < 1e-9


# ── Node Function ──


@pytest.mark.asyncio
async def test_comparison_guard_node_returns_dict():
    """comparison_guard_node must return GraphState-compatible dict."""
    state = {
        "ranked": [
            _make_ranked_product(1, 0.8),
            _make_ranked_product(2, 0.6),
        ],
    }
    result = await comparison_guard_node(state)

    assert isinstance(result, dict)
    assert "comparison" in result
    assert "validated_ranked_ids" in result
    assert result["validated_ranked_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_comparison_guard_node_all_seven_metrics():
    """Node always includes all 7 metrics in diff_matrix regardless of intent."""
    state = {
        "ranked": [
            _make_ranked_product(1, 0.8),
            _make_ranked_product(2, 0.6),
        ],
    }
    result = await comparison_guard_node(state)
    diff_matrix = result["comparison"]["diff_matrix"]
    assert len(diff_matrix) == 7


# ── Exception Hierarchy ──


def test_comparison_integrity_error_is_guard_error():
    """ComparisonIntegrityError must be a subclass of ComparisonGuardError."""
    assert issubclass(ComparisonIntegrityError, ComparisonGuardError)
