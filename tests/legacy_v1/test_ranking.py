"""Unit tests for the deterministic ranking engine.

Verifies:
    1. No LLM imports in ranking.py
    2. Deterministic output (same input → same output)
    3. Secondary sort by product_id ASC for equal scores
    4. Correct weight application per intent profile
    5. Price score computation
    6. All 7 components are included (especially build_score)
"""

from __future__ import annotations

import pytest

from app.services.ranking import (
    rank_candidates,
    compute_price_score,
    compute_popularity_score,
    get_weights_for_intent,
    load_weights,
)


# ── No-LLM Guard ──

def test_no_llm_imports():
    """ranking.py must NEVER import LLM modules."""
    import inspect
    import app.services.ranking as ranking_module

    source = inspect.getsource(ranking_module)
    forbidden = ["openai", "langchain", "anthropic", "google.generativeai"]
    for module in forbidden:
        assert f"import {module}" not in source, f"Forbidden LLM import: {module}"


# ── Deterministic Output ──

def _make_candidate(product_id: int, perf: float = 0.5, battery: float = 0.5,
                    camera: float = 0.5, display: float = 0.5, build: float = 0.5,
                    price: float = 20000) -> dict:
    """Helper to create a test candidate."""
    return {
        "product_id": product_id,
        "product_slug": f"product-{product_id}",
        "brand": "test",
        "model_name": f"Model {product_id}",
        "canonical_title": f"Test Product {product_id}",
        "category": "smartphone",
        "vector_distance": 0.1,
        "specs": {},
        "variants": [{"price_inr": price, "availability": True, "vendor": "test"}],
        "scores": {
            "performance_score": perf,
            "battery_score": battery,
            "camera_score": camera,
            "display_score": display,
            "build_score": build,
        },
    }


def test_deterministic_same_input_same_output():
    """Same candidates → byte-identical ranked output."""
    candidates = [
        _make_candidate(1, perf=0.9, camera=0.8),
        _make_candidate(2, perf=0.7, camera=0.9),
        _make_candidate(3, perf=0.8, camera=0.7),
    ]

    result1 = rank_candidates(candidates)
    result2 = rank_candidates(candidates)

    assert [r["product_id"] for r in result1] == [r["product_id"] for r in result2]
    assert [r["final_score"] for r in result1] == [r["final_score"] for r in result2]


def test_secondary_sort_product_id_asc():
    """Equal scores → sorted by product_id ASC."""
    candidates = [
        _make_candidate(3, perf=0.5, battery=0.5, camera=0.5, display=0.5, build=0.5),
        _make_candidate(1, perf=0.5, battery=0.5, camera=0.5, display=0.5, build=0.5),
        _make_candidate(2, perf=0.5, battery=0.5, camera=0.5, display=0.5, build=0.5),
    ]

    result = rank_candidates(candidates)
    ids = [r["product_id"] for r in result]

    # All scores are equal, so product_id ASC should determine order
    assert ids == [1, 2, 3]


def test_build_score_included():
    """build_score must be part of the ranking formula."""
    # One product with high build_score, another with low
    candidates = [
        _make_candidate(1, build=0.0),
        _make_candidate(2, build=1.0),
    ]

    result = rank_candidates(candidates)
    # Product 2 should rank higher because it has build_score=1.0
    assert result[0]["product_id"] == 2


def test_all_seven_components_in_formula():
    """The ranking formula must use exactly 7 weighted components."""
    weights = get_weights_for_intent([])
    expected_keys = {
        "w_performance", "w_battery", "w_camera",
        "w_display", "w_build", "w_price", "w_popularity"
    }
    assert set(weights.keys()) == expected_keys
    assert abs(sum(weights.values()) - 1.0) < 0.001  # Weights sum to 1.0


# ── Weight Profiles ──

def test_camera_priority_profile():
    """Camera priority should weight w_camera highest."""
    weights = get_weights_for_intent(["camera"])
    assert weights["w_camera"] == max(weights.values())


def test_budget_priority_profile():
    """Budget priority should weight w_price highest."""
    weights = get_weights_for_intent(["budget"])
    assert weights["w_price"] == max(weights.values())


# ── Price Score ──

def test_price_score_with_budget():
    """Cheaper relative to budget → higher score."""
    score_cheap = compute_price_score(15000, budget_max=30000)
    score_expensive = compute_price_score(28000, budget_max=30000)
    assert score_cheap > score_expensive


def test_price_score_no_budget():
    """Without budget, price normalized against reference max."""
    score = compute_price_score(50000)
    assert 0.0 <= score <= 1.0


def test_price_score_none():
    """None price → 0.0 score."""
    assert compute_price_score(None) == 0.0


def test_identical_scores_deterministic_ordering():
    """Identical component scores → identical ranking via product_id tiebreaker.

    Per spec: when final_score is tied, secondary sort by product_id ASC
    must produce consistent output across multiple runs.
    """
    # All products have identical scores
    candidates = [
        _make_candidate(product_id=99, perf=0.7, battery=0.7, camera=0.7, display=0.7, build=0.7, price=30000),
        _make_candidate(product_id=3, perf=0.7, battery=0.7, camera=0.7, display=0.7, build=0.7, price=30000),
        _make_candidate(product_id=50, perf=0.7, battery=0.7, camera=0.7, display=0.7, build=0.7, price=30000),
    ]

    results = []
    for _ in range(10):
        ranked = rank_candidates(candidates, budget_max=50000)
        ids = [r["product_id"] for r in ranked]
        results.append(ids)

    # All 10 runs must produce same order
    for run_ids in results:
        assert run_ids == results[0], f"Non-deterministic: {run_ids} != {results[0]}"

    # With identical scores, product_id ascending is the tiebreaker
    assert results[0] == [3, 50, 99]

