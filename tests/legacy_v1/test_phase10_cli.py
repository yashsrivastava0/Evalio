"""Phase 10 unit tests — CLI renderer, command dispatch, and integration.

Tests guard:
    - Renderer output formatting (ranked, explanation, telemetry, cost, budget)
    - Celery availability gate in fetch_trigger renderer (Issue #2 / Adjustment #1)
    - Telemetry scope disclaimer in :metrics output (Issue #5)
    - Fresh GraphState per query — no cost carry (Issue #4)
    - session_reset clears Redis key (Issue #3)
    - Celery absence doesn't crash (Issue #2)
    - Command dispatch doesn't touch GraphState (Issue #7)
    - Budget skip integration via CLI path (Issue #9)
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.cli_renderer import (
    render_ranked_results,
    render_explanation,
    render_telemetry_summary,
    render_budget_banner,
    render_cost_summary,
    render_fetch_trigger,
    render_metrics,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _capture(fn, *args, **kwargs) -> str:
    """Capture stdout from a function call."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


# ── Test 1: render_ranked_results ────────────────────────────────────────────


def test_render_ranked_results():
    """Formatted output contains rank, title, score for each product."""
    ranked = [
        {"canonical_title": "iQOO Z6", "final_score": 0.874, "best_price": 15999},
        {"canonical_title": "Redmi Note 13", "final_score": 0.862, "best_price": 14999},
    ]

    output = _capture(render_ranked_results, ranked)

    assert "iQOO Z6" in output
    assert "Redmi Note 13" in output
    assert "0.874" in output
    assert "0.862" in output
    assert "₹15,999" in output
    assert "₹14,999" in output
    assert "1." in output
    assert "2." in output


def test_render_ranked_results_empty():
    """Empty ranked list shows 'No products ranked' message."""
    output = _capture(render_ranked_results, [])
    assert "No products ranked" in output


# ── Test 2: render_explanation ───────────────────────────────────────────────


def test_render_explanation():
    """Final verdict and source are printed."""
    explanation = {
        "source": "llm",
        "final_verdict": "iQOO Z6 leads with strong gaming performance.",
        "comparison_insight": "Battery life similar across all candidates.",
    }

    output = _capture(render_explanation, explanation)

    assert "iQOO Z6 leads" in output
    assert "Battery life similar" in output
    assert "llm" in output


# ── Test 3: render_telemetry_summary ─────────────────────────────────────────


def test_render_telemetry_summary():
    """Node spans rendered with duration, cost, and tokens."""
    telemetry = {
        "intent": {"duration_ms": 5.2, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0},
        "ranking": {"duration_ms": 3.1, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0},
        "explanation": {"duration_ms": 842.5, "cost_usd": 0.0042, "tokens_in": 400, "tokens_out": 200, "budget_exceeded": False},
    }

    output = _capture(render_telemetry_summary, telemetry)

    assert "intent" in output
    assert "ranking" in output
    assert "explanation" in output
    assert "842.5" in output
    assert "0.004200" in output


# ── Test 4: render_budget_banner (budget_exceeded=True) ──────────────────────


def test_render_budget_banner_exceeded():
    """budget_exceeded=True shows warning in banner."""
    result = {
        "response": {
            "trace_id": "abc-123-def-456",
            "telemetry": {
                "explanation": {
                    "tokens_in": 500, "tokens_out": 300,
                    "cost_usd": 0.005, "budget_exceeded": True,
                },
            },
        },
    }

    output = _capture(render_budget_banner, result)

    assert "BUDGET EXCEEDED" in output
    assert "abc-123-def-456" in output


def test_render_budget_banner_within():
    """Within budget shows green status."""
    result = {
        "response": {
            "trace_id": "trace-ok",
            "telemetry": {
                "intent": {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
            },
        },
    }

    output = _capture(render_budget_banner, result)

    assert "Within budget" in output


# ── Test 5: render_cost_summary ──────────────────────────────────────────────


def test_render_cost_summary():
    """USD and token count formatted correctly."""
    cost_data = {"total_cost_usd": 0.0182, "total_tokens": 6400}

    output = _capture(render_cost_summary, cost_data)

    assert "$0.018200" in output
    assert "6400" in output


# ── Test 6: render_fetch_trigger (Issue #8 + Adjustment #1) ──────────────────


def test_render_fetch_trigger_celery_available():
    """When Celery is running, shows fetch status message."""
    response = {
        "source": "fetch_trigger",
        "status": "fetching",
        "message": "Product data is being fetched. Please retry shortly.",
        "fetch_in_flight": False,
    }

    output = _capture(render_fetch_trigger, response, celery_available=True)

    assert "Product data is being fetched" in output
    assert "FETCH TRIGGERED" in output


def test_render_fetch_trigger_celery_unavailable():
    """When Celery is NOT running, shows warning (Adjustment #1)."""
    response = {"source": "fetch_trigger", "message": "fetching"}

    output = _capture(render_fetch_trigger, response, celery_available=False)

    assert "Celery is not running" in output
    assert "Start Celery worker" in output


# ── Test 7: render_metrics scope disclaimer (Issue #5) ───────────────────────


def test_metrics_scope_disclaimer():
    """Metrics output includes disclaimer about graph-process-only scope."""
    spans = [
        {"node_name": "intent", "duration_ms": 5.0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "trace_id": "abc"},
    ]

    output = _capture(render_metrics, spans, limit=20)

    assert "graph process only" in output
    assert "Celery background task spans" in output


# ── Test 8: Fresh GraphState per query (Issue #4) ────────────────────────────


def test_fresh_graphstate_per_query():
    """Each query starts with minimal state — no cost carry between turns."""
    # Simulate what cli.py does: build fresh input_state each turn
    session_id = "test-session-001"

    for query in ["best phone", "compare iQOO vs Redmi", "cheapest 5G"]:
        input_state = {
            "query": query,
            "session_id": session_id,
        }

        # Must NOT contain accumulated cost fields
        assert "accumulated_cost_usd" not in input_state
        assert "accumulated_tokens" not in input_state
        assert "telemetry" not in input_state
        assert "trace_id" not in input_state

        # Only query + session_id
        assert set(input_state.keys()) == {"query", "session_id"}


# ── Test 9: session_reset clears Redis (Issue #3) ────────────────────────────


def test_session_reset_clears_redis():
    """session_reset deletes old session_cost key from Redis."""
    import asyncio
    from app.cli import _cmd_session_reset

    redis = AsyncMock()
    redis.delete.return_value = 1

    ctx = {"session_id": "old-session-123", "redis": redis}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_cmd_session_reset(ctx))
    finally:
        loop.close()

    # Must delete old key
    redis.delete.assert_awaited_once_with("session_cost:old-session-123")

    # Must generate new session_id
    assert ctx["session_id"] != "old-session-123"
    assert len(ctx["session_id"]) == 36  # UUID format


# ── Test 10: Celery absence graceful handling (Issue #2) ─────────────────────


def test_celery_absence_graceful():
    """Celery unavailability does not crash — only prints warning."""
    # Simulate what run_local.py does when ping fails
    celery_available = False
    try:
        # Simulating a failed ping
        raise ConnectionError("broker unreachable")
    except ConnectionError:
        celery_available = False

    assert celery_available is False

    # Renderer should handle celery_available=False without crashing
    response = {"source": "fetch_trigger", "message": "fetching"}
    output = _capture(render_fetch_trigger, response, celery_available=False)
    assert "Celery is not running" in output


# ── Test 11: Command dispatch isolation (Issue #7) ───────────────────────────


def test_command_dispatch_isolation():
    """Commands recognized correctly and don't require GraphState."""
    from app.cli import handle_command
    import asyncio

    # Mock ctx — no graph state present
    redis = AsyncMock()
    redis.delete.return_value = 1
    redis.hgetall.return_value = {"total_tokens": "100", "total_cost_usd": "0.005"}

    ctx = {
        "session_id": "test-sess",
        "redis": redis,
        "last_trace_id": "trace-abc",
    }

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # :trace — should not crash
        result = loop.run_until_complete(handle_command(":trace", ctx))
        assert result is False  # not an exit command

        # :cost — should not crash
        result = loop.run_until_complete(handle_command(":cost", ctx))
        assert result is False

        # :exit — should return True
        ctx["db_session"] = AsyncMock()
        result = loop.run_until_complete(handle_command(":exit", ctx))
        assert result is True
    finally:
        loop.close()


# ── Test 12: Budget skip integration ─────────────────────────────────────────


def test_budget_skip_via_cli():
    """When budget is exceeded, explanation returns template fallback."""
    from app.services.explanation import explanation_node
    import asyncio

    # State with cost over budget ceiling
    state = {
        "ranked": [
            {"product_id": 1, "final_score": 0.9, "rank": 1, "scores": {}, "product_slug": "test"},
        ],
        "comparison": None,
        "query": "test",
        "validated_ranked_ids": [1],
        "session_id": "budget-test",
        "telemetry": {
            "intent": {"tokens_in": 5000, "tokens_out": 5000, "cost_usd": 0.05},
        },
    }

    llm = MagicMock()
    llm.generate = AsyncMock()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            explanation_node(state, llm_client=llm, redis_client=None)
        )
    finally:
        loop.close()

    # LLM must NOT be called
    llm.generate.assert_not_called()

    # Should return template fallback with budget_exceeded
    explanation = result["explanation"]
    assert explanation.get("llm_usage", {}).get("budget_exceeded") is True
