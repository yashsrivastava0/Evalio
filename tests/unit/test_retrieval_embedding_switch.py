"""Unit tests — retrieval embedding type selection (Phase 6).

Tests:
    - select_embedding_type returns "specs" by default
    - Returns "review" when priority contains camera/battery keywords
    - Returns "review" when intent_type == "product_qa"
    - Never imports any LLM module in retrieval.py
    - retrieval_node reads embedding_type from select_embedding_type, not from state
    - retrieval_node sets embedding_type in returned state
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.retrieval import select_embedding_type


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _intent(intent_type: str = "search", priorities: list[str] | None = None):
    obj = MagicMock()
    obj.intent_type = intent_type
    obj.priority = priorities or []
    obj.category = "smartphones"
    obj.budget_min = None
    obj.budget_max = None
    obj.brands = []
    obj.min_ram = None
    obj.min_storage = None
    return obj


# ── select_embedding_type ─────────────────────────────────────────────────────


def test_default_returns_specs():
    """No priorities → should return 'specs'."""
    intent = _intent()
    assert select_embedding_type(intent) == "specs"


def test_none_intent_returns_specs():
    """None intent → fallback to 'specs' without error."""
    assert select_embedding_type(None) == "specs"


def test_camera_priority_returns_review():
    """'camera' in priority → 'review' embedding."""
    intent = _intent(priorities=["camera"])
    assert select_embedding_type(intent) == "review"


def test_battery_life_priority_returns_review():
    """'battery life' in priority → 'review' embedding."""
    intent = _intent(priorities=["battery life"])
    assert select_embedding_type(intent) == "review"


def test_battery_priority_returns_review():
    """'battery' keyword triggers review embedding."""
    intent = _intent(priorities=["battery"])
    assert select_embedding_type(intent) == "review"


def test_heating_priority_returns_review():
    """'heating' keyword triggers review embedding."""
    intent = _intent(priorities=["heating"])
    assert select_embedding_type(intent) == "review"


def test_durability_priority_returns_review():
    """'durability' keyword triggers review embedding."""
    intent = _intent(priorities=["durability"])
    assert select_embedding_type(intent) == "review"


def test_review_keyword_returns_review():
    """'review' keyword triggers review embedding."""
    intent = _intent(priorities=["review"])
    assert select_embedding_type(intent) == "review"


def test_product_qa_intent_returns_review():
    """intent_type='product_qa' must use reviews embedding."""
    intent = _intent(intent_type="product_qa")
    assert select_embedding_type(intent) == "review"


def test_search_intent_with_non_review_priorities_returns_specs():
    """Non-review priorities and search intent → specs."""
    intent = _intent(intent_type="search", priorities=["price", "storage"])
    assert select_embedding_type(intent) == "specs"


def test_mixed_priorities_with_camera_returns_review():
    """Mixed priorities including camera → review."""
    intent = _intent(priorities=["price", "camera", "storage"])
    assert select_embedding_type(intent) == "review"


def test_case_insensitive_priority_matching():
    """Priority keywords should be case-insensitive."""
    intent = _intent(priorities=["Camera"])
    assert select_embedding_type(intent) == "review"

    intent2 = _intent(priorities=["BATTERY"])
    assert select_embedding_type(intent2) == "review"


# ── No LLM in retrieval ─────────────────────────────────────────────────────


def test_retrieval_has_no_llm_import():
    """retrieval.py must not import any LLM client."""
    import app.services.retrieval as module
    source_file = __import__("inspect").getfile(module)
    with open(source_file) as f:
        source = f.read()
    assert "from langchain" not in source
    assert "import openai" not in source
    assert "ChatOpenAI" not in source
    assert "ChatGoogleGenerativeAI" not in source


def test_select_embedding_type_has_no_await():
    """select_embedding_type must be a synchronous function (no async, no IO)."""
    import inspect
    assert not inspect.iscoroutinefunction(select_embedding_type)


# ── retrieval_node state output ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_node_returns_embedding_type_in_state():
    """retrieval_node must include 'embedding_type' in its state return.

    camera priority → review type selected. Review embeddings exist (non-empty)
    so fallback is NOT triggered. State must reflect 'review'.
    """
    from unittest.mock import AsyncMock, patch
    from app.services.retrieval import retrieval_node

    mock_db = AsyncMock()
    state = {
        "intent": _intent(priorities=["camera"]),
        "query_embedding": [0.1] * 1536,
        "use_session_comparison": False,
    }

    # Return non-empty results for review type — fallback must NOT fire
    review_results = [{"product_id": 1, "distance": 0.1}, {"product_id": 2, "distance": 0.2}]
    candidates_output = [{"product_id": 1}, {"product_id": 2}]

    with patch("app.services.retrieval.sql_filter_candidates", new_callable=AsyncMock, return_value=[1, 2]), \
         patch("app.services.retrieval.vector_rerank_candidates", new_callable=AsyncMock, return_value=review_results), \
         patch("app.services.retrieval.reduce_candidates", new_callable=AsyncMock, return_value=candidates_output):

        result = await retrieval_node(state, db_session=mock_db)

    assert "embedding_type" in result
    assert result["embedding_type"] == "review"  # camera priority → review, no fallback


@pytest.mark.asyncio
async def test_retrieval_node_falls_back_to_specs_when_review_empty():
    """retrieval_node falls back to 'specs' when review embedding returns no results."""
    from unittest.mock import AsyncMock, patch, call
    from app.services.retrieval import retrieval_node

    mock_db = AsyncMock()
    state = {
        "intent": _intent(priorities=["camera"]),
        "query_embedding": [0.1] * 1536,
        "use_session_comparison": False,
    }

    # First call (review) returns empty; second call (specs fallback) returns results
    specs_results = [{"product_id": 5, "distance": 0.05}]
    vec_mock = AsyncMock(side_effect=[[], specs_results])   # review empty → specs hit

    with patch("app.services.retrieval.sql_filter_candidates", new_callable=AsyncMock, return_value=[5]), \
         patch("app.services.retrieval.vector_rerank_candidates", vec_mock), \
         patch("app.services.retrieval.reduce_candidates", new_callable=AsyncMock, return_value=[{"product_id": 5}]):

        result = await retrieval_node(state, db_session=mock_db)

    # After fallback, embedding_type must be updated to "specs"
    assert result["embedding_type"] == "specs"


@pytest.mark.asyncio
async def test_retrieval_node_defaults_embedding_type_to_specs():
    """retrieval_node returns 'specs' embedding_type by default."""
    from unittest.mock import AsyncMock, patch
    from app.services.retrieval import retrieval_node

    mock_db = AsyncMock()
    state = {
        "intent": _intent(),
        "query_embedding": [],
        "use_session_comparison": False,
    }

    with patch("app.services.retrieval.sql_filter_candidates", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.retrieval.vector_rerank_candidates", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.retrieval.reduce_candidates", new_callable=AsyncMock, return_value=[]):

        result = await retrieval_node(state, db_session=mock_db)

    assert result.get("embedding_type") == "specs"
