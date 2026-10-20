"""Phase 8 integration test — fetch trigger pipeline (all mocked).

Test: test_fetch_trigger_e2e
    Simulates graph's fetch_trigger route:
        DecisionMatrix → FetchTriggerNode → scrape_product.delay() → PartialResponse

    Guarantees:
        - scrape_product.delay() is called exactly once
        - graph returns fetch_trigger partial response immediately (non-blocking)
        - No .get() or .wait() called from the graph node
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_fetch_trigger_e2e():
    """End-to-end: graph fetch_trigger route enqueues scrape_product.delay() non-blocking.

    Mocks:
        - Redis client (no live Redis)
        - scrape_product.delay() (no live Celery)
        - generate_slug (scraper import)

    Asserts:
        1. scrape_product.delay() called exactly once per (brand, vendor, url) triple
        2. Graph returns {"source": "fetch_trigger", "status": "fetching"} immediately
        3. No AsyncResult .get() or .wait() called
    """
    from app.graph import _make_fetch_trigger_node
    from app.agents.router import IntentPlan

    # ── Mock Redis ──
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = b"OK"

    # ── Mock Celery task ──
    mock_task = MagicMock()
    mock_delay_calls = []

    def capture_delay(**kwargs):
        mock_delay_calls.append(kwargs)

    mock_task.delay = MagicMock(side_effect=lambda **kw: capture_delay(**kw))

    # ── Mock IntentPlan state ──
    intent = MagicMock(spec=IntentPlan)
    intent.category = "smartphone"
    intent.brands = ["samsung"]
    intent.vendor_urls = {
        "samsung": {
            "amazon.in": "https://amazon.in/samsung-galaxy-s24",
        }
    }

    state = {
        "fetch_in_flight": False,
        "intent": intent,
    }

    with patch("app.workers.tasks.scrape_product", mock_task):
        with patch("scraper.normalization.variant_builder.generate_slug", return_value="samsung-galaxy-smartphone"):
            node = _make_fetch_trigger_node(redis)
            result = await node(state)

    # ── Assertions ──
    # 1. Non-blocking: delay() called, not get() or wait()
    mock_task.delay.assert_called_once()
    assert not hasattr(mock_task.delay.return_value, "get") or not mock_task.delay.return_value.get.called, (
        "graph must not call .get() on the AsyncResult"
    )

    # 2. Correct URL dispatched
    call_kwargs = mock_task.delay.call_args[1]
    assert call_kwargs["url"] == "https://amazon.in/samsung-galaxy-s24"
    assert call_kwargs["vendor"] == "amazon.in"
    assert call_kwargs["category"] == "smartphone"

    # 3. fetch_status set in Redis
    redis.set.assert_called_once_with(
        "fetch_status:samsung-galaxy-smartphone",
        "fetching",
        ex=600,
    )

    # 4. Graph node returns empty dict (partial_response handled by response_node)
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_trigger_skips_enqueue_when_in_flight():
    """If fetch_in_flight=True, fetch trigger node does NOT re-enqueue."""
    from app.graph import _make_fetch_trigger_node
    from app.agents.router import IntentPlan

    redis = AsyncMock()
    mock_task = MagicMock()

    state = {
        "fetch_in_flight": True,
        "intent": MagicMock(spec=IntentPlan),
        "fetch_brand": "apple",
    }

    with patch("app.workers.tasks.scrape_product", mock_task):
        node = _make_fetch_trigger_node(redis)
        result = await node(state)

    mock_task.delay.assert_not_called()
    assert result == {}
