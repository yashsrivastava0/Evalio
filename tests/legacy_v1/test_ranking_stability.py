"""Unit tests for ranking stability and determinism.

Verifies:
    1. 100-run determinism across varied candidate sizes
    2. Ranking uses ONLY the 7 canonical scores
    3. product_id ASC tiebreaker for identical scores
    4. No LLM involvement in ranking module
    5. Stable output for shuffled input order
"""

from __future__ import annotations

import inspect
import random

import pytest

from app.services.ranking import rank_candidates, get_weights_for_intent


# ── Helper ──

def _make_candidate(product_id: int, **score_overrides) -> dict:
    """Build a test candidate with default scores."""
    scores = {
        "performance_score": 0.5,
        "battery_score": 0.5,
        "camera_score": 0.5,
        "display_score": 0.5,
        "build_score": 0.5,
        "price_score": 0.5,
        "popularity_score": 0.5,
    }
    scores.update(score_overrides)

    return {
        "product_id": product_id,
        "product_slug": f"product-{product_id}",
        "brand": "test",
        "model_name": f"Model {product_id}",
        "canonical_title": f"Test Product {product_id}",
        "category": "smartphone",
        "vector_distance": 0.1,
        "specs": {},
        "variants": [{"price_inr": 30000, "availability": True, "vendor": "test"}],
        "scores": scores,
    }


# ── 100-Run Determinism ──

@pytest.mark.parametrize("candidate_count", [10, 20, 50, 100])
def test_100_run_determinism(candidate_count: int):
    """Ranking must produce identical output across 100 runs for any candidate count."""
    random.seed(42)
    candidates = [
        _make_candidate(
            i,
            performance_score=round(random.uniform(0.1, 1.0), 4),
            battery_score=round(random.uniform(0.1, 1.0), 4),
            camera_score=round(random.uniform(0.1, 1.0), 4),
            display_score=round(random.uniform(0.1, 1.0), 4),
            build_score=round(random.uniform(0.1, 1.0), 4),
        )
        for i in range(1, candidate_count + 1)
    ]

    first_result = rank_candidates(candidates, budget_max=60_000)
    first_ids = [r["product_id"] for r in first_result]
    first_scores = [r["final_score"] for r in first_result]

    for run in range(99):
        result = rank_candidates(candidates, budget_max=60_000)
        ids = [r["product_id"] for r in result]
        scores = [r["final_score"] for r in result]
        assert ids == first_ids, f"Determinism failure at run {run + 2}: {ids} != {first_ids}"
        assert scores == first_scores, f"Score mismatch at run {run + 2}"


# ── 7 Canonical Scores Only ──

def test_ranking_uses_only_seven_canonical_scores():
    """Weight profiles must contain exactly the 7 canonical score keys."""
    weights = get_weights_for_intent([])
    expected = {
        "w_performance", "w_battery", "w_camera",
        "w_display", "w_build", "w_price", "w_popularity",
    }
    assert set(weights.keys()) == expected, \
        f"Unexpected weight keys: {set(weights.keys()) - expected}"


def test_all_weight_profiles_have_seven_keys():
    """Every weight profile must have exactly 7 keys summing to ~1.0."""
    from app.services.ranking import load_weights
    all_weights = load_weights()

    for profile_name, weights in all_weights.items():
        assert set(weights.keys()) == {
            "w_performance", "w_battery", "w_camera",
            "w_display", "w_build", "w_price", "w_popularity",
        }, f"Profile '{profile_name}' has wrong keys: {set(weights.keys())}"

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, \
            f"Profile '{profile_name}' weights sum to {total}, expected ~1.0"


def test_ranking_formula_source_uses_seven_components():
    """rank_candidates source must reference all 7 score components."""
    source = inspect.getsource(rank_candidates)
    required_scores = [
        "performance_score", "battery_score", "camera_score",
        "display_score", "build_score", "price_score", "popularity_score",
    ]
    for score in required_scores:
        assert score in source, f"Missing {score} in ranking formula"


# ── Tiebreaker ──

def test_tiebreaker_product_id_ascending():
    """Identical final scores → product_id ASC ordering."""
    candidates = [
        _make_candidate(99),
        _make_candidate(3),
        _make_candidate(42),
        _make_candidate(7),
        _make_candidate(15),
    ]

    result = rank_candidates(candidates, budget_max=50_000)
    ids = [r["product_id"] for r in result]

    # All have identical scores, so sort by product_id ASC
    assert ids == sorted(ids), f"Tiebreaker failed: {ids} should be ascending"


def test_tiebreaker_stable_with_shuffled_input():
    """Shuffled input order must still produce same ranked output."""
    candidates = [
        _make_candidate(99),
        _make_candidate(3),
        _make_candidate(42),
    ]

    result1 = rank_candidates(candidates, budget_max=50_000)
    ids1 = [r["product_id"] for r in result1]

    # Shuffle input
    shuffled = [candidates[2], candidates[0], candidates[1]]
    result2 = rank_candidates(shuffled, budget_max=50_000)
    ids2 = [r["product_id"] for r in result2]

    assert ids1 == ids2, "Ranking must be stable regardless of input order"


# ── No LLM in Ranking ──

def test_no_llm_imports_in_ranking():
    """ranking.py must never import LLM modules."""
    import app.services.ranking as ranking_module
    source = inspect.getsource(ranking_module)

    forbidden = ["openai", "langchain", "anthropic", "google.generativeai", "litellm"]
    for module in forbidden:
        assert f"import {module}" not in source, f"Forbidden LLM import: {module}"


def test_no_llm_calls_in_ranking():
    """ranking.py must not contain any LLM client calls."""
    import app.services.ranking as ranking_module
    source = inspect.getsource(ranking_module)

    forbidden_patterns = ["client.chat", "openai.ChatCompletion", "model.generate", "llm("]
    for pattern in forbidden_patterns:
        assert pattern not in source, f"Forbidden LLM call pattern: {pattern}"
