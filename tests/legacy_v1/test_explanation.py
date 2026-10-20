"""Unit tests for the Explanation service — updated for Phase 7 contract.

Tests:
    1. Template explanation has all 4 required fields (new contract)
    2. ranked_list block structure (product_id, total_score, match_score, one_line_verdict)
    3. pros_cons keyed by product_id string
    4. verdict mentions top product and score
    5. Empty ranked list safe
    6. Prompt does not include weight names
    7. Pre-LLM assertion: sorted by final_score DESC
    8. Pre-LLM assertion: ranked_ids must match validated_ranked_ids
    9. explanation_node returns GraphState-compatible dict with new contract keys
"""

from __future__ import annotations

import pytest

from app.services.explanation import (
    _template_explanation,
    explanation_node,
    build_explanation_prompt,
)


# ── Test Helpers ──


def _make_ranked(product_id: int, slug: str, score: float, rank: int) -> dict:
    """Helper to create a ranked product."""
    return {
        "product_id": product_id,
        "product_slug": slug,
        "canonical_title": f"Test Product {product_id}",
        "final_score": score,
        "rank": rank,
        "scores": {
            "performance_score": 0.8,
            "battery_score": 0.6,
            "camera_score": 0.7,
            "display_score": 0.65,
            "build_score": 0.5,
            "price_score": 0.55,
            "popularity_score": 0.45,
        },
        "best_price": 25000,
    }


# ── Structured Output (Phase 7 Contract) ──


def test_template_explanation_has_required_keys():
    """Template explanation must return all 4 Phase 7 structured contract fields."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    result = _template_explanation(ranked)

    assert "ranked_list" in result
    assert "comparison_highlights" in result
    assert "pros_cons" in result
    assert "final_verdict" in result
    assert "source" in result
    assert result["source"] == "template"


def test_template_ranked_list_block_structure():
    """ranked_list must contain product_id, total_score, match_score, one_line_verdict."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    result = _template_explanation(ranked)
    block = result["ranked_list"]

    assert len(block) == 1
    item = block[0]
    assert item["product_id"] == 1
    assert abs(item["total_score"] - 0.85) < 0.001
    assert "match_score" in item
    assert "one_line_verdict" in item
    assert "0.850" in item["one_line_verdict"]
    assert "#1" in item["one_line_verdict"]


def test_template_pros_cons_keyed_by_product_id_str():
    """pros_cons must be a dict keyed by product_id as string."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    result = _template_explanation(ranked)
    pc = result["pros_cons"]

    assert "1" in pc
    assert isinstance(pc["1"]["pros"], list)
    assert isinstance(pc["1"]["cons"], list)


def test_template_verdict_mentions_top_product():
    """Verdict must reference the #1 product title and score."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    result = _template_explanation(ranked)

    assert "Test Product 1" in result["final_verdict"]
    assert "0.850" in result["final_verdict"]


def test_template_empty_ranked():
    """Empty ranked list should produce 'No products to rank' verdict."""
    result = _template_explanation([])

    assert result["final_verdict"] == "No products to rank."
    assert result["ranked_list"] == []
    assert result["pros_cons"] == {}


# ── Prompt Safety ──


def test_prompt_does_not_include_weights():
    """LLM prompt must NOT include weight values or ranking math."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    prompt = build_explanation_prompt(ranked, query="best phone")

    assert "w_performance" not in prompt
    assert "w_battery" not in prompt
    assert "w_camera" not in prompt
    # Scores are allowed — they're pre-filled numeric data
    assert "Performance:" in prompt


def test_prompt_restricts_llm_from_ranked_list():
    """Prompt must instruct LLM NOT to return ranked_list."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    prompt = build_explanation_prompt(ranked, query="best phone")
    assert "ranked_list" in prompt.lower()


# ── Node Assertions ──


@pytest.mark.asyncio
async def test_explanation_node_pre_assertion_sorted():
    """Pre-LLM assertion: ranked list must be sorted by final_score DESC."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    result = await explanation_node({
        "ranked": ranked,
        "comparison": None,
        "query": "test query",
        "validated_ranked_ids": [1, 2],
    })

    assert "explanation" in result
    assert isinstance(result["explanation"], dict)


@pytest.mark.asyncio
async def test_explanation_node_pre_assertion_unsorted_fails():
    """Pre-LLM assertion must fail if ranked list is NOT sorted."""
    ranked = [
        _make_ranked(2, "phone-b", 0.70, 2),
        _make_ranked(1, "phone-a", 0.85, 1),
    ]
    with pytest.raises(AssertionError, match="not sorted"):
        await explanation_node({
            "ranked": ranked,
            "comparison": None,
            "query": "test query",
            "validated_ranked_ids": [],
        })


@pytest.mark.asyncio
async def test_explanation_node_id_mismatch_fails():
    """Pre-LLM assertion must fail if ranked_ids don't match validated_ranked_ids."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    with pytest.raises(AssertionError, match="ranked_ids mismatch"):
        await explanation_node({
            "ranked": ranked,
            "comparison": None,
            "query": "test query",
            "validated_ranked_ids": [2, 1],  # Wrong order
        })


@pytest.mark.asyncio
async def test_explanation_node_returns_phase7_contract():
    """explanation_node output must have Phase 7 contract fields."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    result = await explanation_node({
        "ranked": ranked,
        "comparison": None,
        "query": "test query",
        "validated_ranked_ids": [1, 2],
    })

    explanation = result["explanation"]
    assert "ranked_list" in explanation
    assert "comparison_highlights" in explanation
    assert "pros_cons" in explanation
    assert "final_verdict" in explanation
