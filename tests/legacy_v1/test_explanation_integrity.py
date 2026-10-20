"""Phase 7 — Explanation integrity unit tests.

Verifies all locked LLM containment invariants:
    - ranked_list always comes from template, never LLM
    - LLM cannot modify numeric values (total_score, match_score)
    - LLM cannot control winner field in comparison_highlights
    - one_line_verdict is always template-generated
    - All 4 contract fields present in every output
    - LLM timeout falls back to template
    - LLM containment violation (ranked_list in response) falls back to template
    - Output product ID order always matches ranked_ids
    - Correct merge strategy: template preserved, LLM prose merged by key
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.explanation import (
    _template_explanation,
    _merge_llm_prose,
    build_explanation_prompt,
    explanation_node,
    generate_explanation,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ranked(product_id: int, slug: str, score: float, rank: int) -> dict:
    return {
        "product_id": product_id,
        "product_slug": slug,
        "canonical_title": f"Product {product_id}",
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


def _make_llm_client(response: dict | str | None = None, *, raise_exc=None, timeout=False):
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.model = "mock-llm"
    if timeout:
        async def _gen(*args, **kwargs):
            await asyncio.sleep(60)  # exceeds 30s timeout
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = _gen
    elif raise_exc:
        async def _gen_err(*args, **kwargs):
            raise raise_exc
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = _gen_err
    else:
        payload = json.dumps(response) if isinstance(response, dict) else (response or "")
        mock_choice = MagicMock()
        mock_choice.message.content = payload
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 10
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_completion)
    return client


# ── Template Contract ─────────────────────────────────────────────────────────


def test_explanation_contract_all_fields_present():
    """Template must return all 4 mandatory contract fields."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1), _make_ranked(2, "phone-b", 0.70, 2)]
    result = _template_explanation(ranked)
    assert "ranked_list" in result
    assert "comparison_highlights" in result
    assert "pros_cons" in result
    assert "final_verdict" in result


def test_ranked_list_always_from_template():
    """ranked_list must be built from template data — all fields must be numeric/strings from input."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    result = _template_explanation(ranked)
    item = result["ranked_list"][0]
    assert item["product_id"] == 1
    assert abs(item["total_score"] - 0.85) < 1e-9    # from final_score
    assert isinstance(item["one_line_verdict"], str)
    assert "1" in item["one_line_verdict"]  # rank present
    assert "0.850" in item["one_line_verdict"]  # score present


def test_one_line_verdict_is_template_generated():
    """one_line_verdict must be deterministic — contains rank and score, not LLM prose."""
    ranked = [_make_ranked(1, "phone-a", 0.923, 1)]
    result = _template_explanation(ranked)
    verdict = result["ranked_list"][0]["one_line_verdict"]
    assert "0.923" in verdict
    assert "#1" in verdict


def test_template_numeric_values_frozen():
    """Numeric values in ranked_list must exactly match input — no rounding mutation."""
    ranked = [_make_ranked(1, "phone-a", 0.12345678, 1)]
    result = _template_explanation(ranked)
    # total_score is taken directly from final_score (float conversion only)
    assert abs(result["ranked_list"][0]["total_score"] - 0.12345678) < 1e-9


def test_template_pros_cons_by_product_id():
    """pros_cons must be keyed by product_id string."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1), _make_ranked(2, "phone-b", 0.70, 2)]
    result = _template_explanation(ranked)
    assert "1" in result["pros_cons"]
    assert "2" in result["pros_cons"]
    assert isinstance(result["pros_cons"]["1"]["pros"], list)
    assert isinstance(result["pros_cons"]["1"]["cons"], list)


def test_template_empty_ranked():
    """Empty ranked list must produce safe defaults."""
    result = _template_explanation([])
    assert result["ranked_list"] == []
    assert result["final_verdict"] == "No products to rank."


# ── LLM Containment ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_cannot_modify_numeric_values():
    """LLM prose must not affect total_score, match_score, or one_line_verdict in output."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1), _make_ranked(2, "phone-b", 0.70, 2)]
    template = _template_explanation(ranked)
    original_score = template["ranked_list"][0]["total_score"]
    original_verdict = template["ranked_list"][0]["one_line_verdict"]

    llm_client = _make_llm_client({
        "pros_cons": {"1": {"pros": ["Great battery"], "cons": ["Expensive"]}},
        "comparison_highlights": [],
        "final_verdict": "Phone A is the best choice.",
    })

    result = await generate_explanation(ranked, llm_client=llm_client)

    # Numeric values unchanged
    assert abs(result["ranked_list"][0]["total_score"] - original_score) < 1e-9
    assert result["ranked_list"][0]["one_line_verdict"] == original_verdict


@pytest.mark.asyncio
async def test_llm_cannot_control_winner_field():
    """winner field in comparison_highlights must come from template only."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1), _make_ranked(2, "phone-b", 0.70, 2)]
    comparison = {
        "diff_matrix": {
            "performance_score": {"absolute_diff": 0.2, "percent_diff": 25.0, "winner": "1"},
            **{m: {"absolute_diff": 0.0, "percent_diff": 0.0, "winner": None}
               for m in ["battery_score", "camera_score", "display_score",
                         "build_score", "price_score", "popularity_score"]},
        }
    }

    # LLM tries to return a fake comparison_highlights with wrong winner
    llm_client = _make_llm_client({
        "pros_cons": {},
        "comparison_highlights": [
            {"metric": "performance_score", "explanation_reason_text": "Both are equal"}
        ],
        "final_verdict": "Hard to say.",
    })

    result = await generate_explanation(ranked, comparison=comparison, llm_client=llm_client)

    # winner must still be "1" from template — not overridden by LLM
    perf_h = next(
        h for h in result["comparison_highlights"] if h["metric"] == "performance_score"
    )
    assert perf_h["winner"] == "1"
    assert perf_h["numeric_reason"] is not None and "25" in str(perf_h["numeric_reason"])


@pytest.mark.asyncio
async def test_llm_ranked_list_injection_rejected():
    """If LLM returns 'ranked_list', the response must be rejected and template used."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]

    llm_client = _make_llm_client({
        "ranked_list": [{"product_id": 999, "total_score": 9.9}],  # containment violation
        "pros_cons": {},
        "comparison_highlights": [],
        "final_verdict": "Attempted injection.",
    })

    result = await generate_explanation(ranked, llm_client=llm_client)

    # Must fall back — ranked_list from template (product_id=1, not 999)
    assert result["ranked_list"][0]["product_id"] == 1
    assert "fallback" in result.get("source", "")


@pytest.mark.asyncio
async def test_llm_timeout_falls_back_to_template():
    """LLM timeout (>3s) must fall back to template explanation."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    llm_client = _make_llm_client(timeout=True)

    result = await generate_explanation(ranked, llm_client=llm_client)

    assert result["source"] == "template_llm_timeout_fallback"
    assert result["ranked_list"][0]["product_id"] == 1


@pytest.mark.asyncio
async def test_llm_exception_falls_back_to_template():
    """Any LLM exception must fall back to template."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    llm_client = _make_llm_client(raise_exc=RuntimeError("model overloaded"))

    result = await generate_explanation(ranked, llm_client=llm_client)
    assert "template" in result.get("source", "")


# ── Merge Strategy ────────────────────────────────────────────────────────────


def test_merge_does_not_drop_metrics():
    """Merge must preserve all template metrics even if LLM returns partial highlights."""
    ranked = [_make_ranked(1, "a", 0.9, 1), _make_ranked(2, "b", 0.7, 2)]
    comparison = {
        "diff_matrix": {
            m: {"absolute_diff": 0.1, "percent_diff": 10.0, "winner": "1"}
            for m in ["performance_score", "battery_score", "camera_score",
                      "display_score", "build_score", "price_score", "popularity_score"]
        }
    }
    template = _template_explanation(ranked, comparison)

    # LLM only returns prose for 2 metrics
    llm_prose = {
        "pros_cons": {},
        "comparison_highlights": [
            {"metric": "battery_score", "explanation_reason_text": "Phone A has much better battery"}
        ],
        "final_verdict": "Phone A wins.",
    }

    merged = _merge_llm_prose(template, llm_prose)

    # All 7 metrics must still be present
    metrics_in_output = {h["metric"] for h in merged["comparison_highlights"]}
    for m in ["performance_score", "battery_score", "camera_score",
              "display_score", "build_score", "price_score", "popularity_score"]:
        assert m in metrics_in_output, f"Metric {m} dropped from comparison_highlights"


def test_merge_explanation_reason_text_only_where_provided():
    """explanation_reason_text must only be set for metrics LLM provided."""
    ranked = [_make_ranked(1, "a", 0.9, 1)]
    template = _template_explanation(ranked)
    template["comparison_highlights"] = [
        {"metric": "battery_score", "winner": "1", "numeric_reason": "X leads by 0.1 (10%)", "explanation_reason_text": ""},
        {"metric": "camera_score", "winner": "1", "numeric_reason": "X leads by 0.2 (20%)", "explanation_reason_text": ""},
    ]

    llm_prose = {
        "pros_cons": {},
        "comparison_highlights": [
            {"metric": "battery_score", "explanation_reason_text": "Great battery life."}
        ],
        "final_verdict": "Good phone.",
    }
    merged = _merge_llm_prose(template, llm_prose)

    battery_h = next(h for h in merged["comparison_highlights"] if h["metric"] == "battery_score")
    camera_h = next(h for h in merged["comparison_highlights"] if h["metric"] == "camera_score")
    assert battery_h["explanation_reason_text"] == "Great battery life."
    assert camera_h["explanation_reason_text"] == ""  # LLM did not provide this


# ── Explanation Node Assertions ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_explanation_ids_match_ranked_ids():
    """ranked_list product_ids in output must match input ranked_ids in order."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    result = await explanation_node({
        "ranked": ranked,
        "comparison": None,
        "query": "best phone",
        "validated_ranked_ids": [1, 2],
    })
    output_ids = [r["product_id"] for r in result["explanation"]["ranked_list"]]
    assert output_ids == [1, 2]


@pytest.mark.asyncio
async def test_explanation_pre_assertion_unsorted_fails():
    """Pre-LLM assertion: unsorted ranked list must raise AssertionError."""
    ranked = [
        _make_ranked(2, "phone-b", 0.70, 2),
        _make_ranked(1, "phone-a", 0.85, 1),  # wrong order
    ]
    with pytest.raises(AssertionError, match="not sorted"):
        await explanation_node({
            "ranked": ranked,
            "comparison": None,
            "query": "test",
            "validated_ranked_ids": [],
        })


@pytest.mark.asyncio
async def test_explanation_pre_assertion_id_mismatch_fails():
    """Pre-LLM assertion: ranked_ids not matching validated_ranked_ids must fail."""
    ranked = [
        _make_ranked(1, "phone-a", 0.85, 1),
        _make_ranked(2, "phone-b", 0.70, 2),
    ]
    with pytest.raises(AssertionError, match="ranked_ids mismatch"):
        await explanation_node({
            "ranked": ranked,
            "comparison": None,
            "query": "test",
            "validated_ranked_ids": [2, 1],  # wrong order
        })


# ── Prompt Safety ─────────────────────────────────────────────────────────────


def test_prompt_does_not_include_weight_variables():
    """LLM prompt must not expose internal weight variable names."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    prompt = build_explanation_prompt(ranked, query="best phone")
    assert "w_performance" not in prompt
    assert "w_battery" not in prompt
    assert "w_camera" not in prompt


def test_prompt_instructs_no_ranked_list():
    """Prompt must explicitly instruct LLM NOT to include ranked_list."""
    ranked = [_make_ranked(1, "phone-a", 0.85, 1)]
    prompt = build_explanation_prompt(ranked, query="best phone")
    assert "ranked_list" in prompt.lower()
    # Should say DON'T include it
    assert "do not include" in prompt.lower() or "not include" in prompt.lower()
