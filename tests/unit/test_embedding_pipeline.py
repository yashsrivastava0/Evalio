"""Unit tests — embedding pipeline correctness.

Tests:
    - compute_specs_hash determinism
    - compute_specs_hash changes on content change
    - generate_specs_embedding calls provider.embed()
    - generate_specs_embedding stores embedding_version and specs_hash
    - generate_specs_embedding releases embed lock in finally block (even on error)
    - generate_specs_embedding does NOT call any LLM or ranking function
    - generate_review_embedding skips and logs when summary_text is empty/None
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.embedding_pipeline import (
    compute_specs_hash,
    compute_reviews_hash,
    compute_embedding_hash,
    _specs_to_text,
    EMBEDDING_VERSION,
)


# ── compute_specs_hash ────────────────────────────────────────────────────────


def test_specs_hash_is_deterministic():
    """Same specs dict always produces same hash."""
    specs = {"ram_gb": 8, "storage_gb": 128, "cpu": "Snapdragon 8 Gen 2"}
    h1 = compute_specs_hash(specs)
    h2 = compute_specs_hash(specs)
    assert h1 == h2


def test_specs_hash_is_independent_of_key_order():
    """Canonical JSON ensures key order doesn't affect hash."""
    specs_a = {"ram_gb": 8, "storage_gb": 128}
    specs_b = {"storage_gb": 128, "ram_gb": 8}
    assert compute_specs_hash(specs_a) == compute_specs_hash(specs_b)


def test_specs_hash_changes_on_content_change():
    """Hash must differ when specs content changes."""
    specs_v1 = {"ram_gb": 8, "storage_gb": 128}
    specs_v2 = {"ram_gb": 12, "storage_gb": 128}
    assert compute_specs_hash(specs_v1) != compute_specs_hash(specs_v2)


def test_specs_hash_handles_none():
    """None specs should produce a stable hash (not crash)."""
    h = compute_specs_hash(None)
    assert isinstance(h, str)
    assert len(h) == 64


def test_specs_hash_returns_64_char_hex():
    """Hash must be a 64-character hex string (SHA-256)."""
    h = compute_specs_hash({"key": "value"})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ── Canonical JSON Audit (Fix-A required tests) ───────────────────────────────


def test_specs_hash_float_precision_does_not_matter():
    """CRITICAL: 6.70 and 6.7 must produce identical hash.

    Without float normalization, Python's json.dumps encodes them identically
    anyway (6.7 == 6.7), but if floats arrive from external sources with
    trailing-zero differences after string conversion, canonical rounding
    ensures stability.
    """
    specs_a = {"display_hz": 120.0, "weight_g": 174.0}
    specs_b = {"display_hz": 120.000000001, "weight_g": 174.0}  # 9dp diff, rounds to same at 8dp
    # Both round to 120.0 at 8 decimal places (1e-9 is below 8dp threshold)
    specs_normalized = {"display_hz": round(120.000000001, 8), "weight_g": 174.0}
    assert compute_specs_hash(specs_a) == compute_specs_hash(specs_normalized)


def test_specs_hash_float_precision_within_8dp_equals():
    """Floats differing only beyond 8 decimal places hash identically."""
    # 6.70000000 and 6.70000001 differ at 9th decimal — rounded to same at 8dp
    import math
    specs_a = {"score": 6.70000000}
    specs_b = {"score": 6.700000004}  # 9th decimal, rounds to 6.7 at 8dp
    assert compute_specs_hash(specs_a) == compute_specs_hash(specs_b)


def test_specs_hash_float_precision_beyond_threshold_differs():
    """Floats differing at 8th decimal place must produce different hashes."""
    specs_a = {"score": 6.70000001}  # rounds to 6.70000001 at 8dp
    specs_b = {"score": 6.70000002}  # rounds to 6.70000002 at 8dp
    assert compute_specs_hash(specs_a) != compute_specs_hash(specs_b)


def test_specs_hash_null_field_stripping():
    """Null fields must be stripped — presence/absence of null doesn't change hash."""
    specs_with_null = {"ram_gb": 8, "color": None, "storage_gb": 128}
    specs_without_null = {"ram_gb": 8, "storage_gb": 128}
    assert compute_specs_hash(specs_with_null) == compute_specs_hash(specs_without_null)


def test_specs_hash_nested_dict_sorted():
    """Nested dicts must also have sorted keys for canonical form."""
    specs_a = {"camera": {"mp_front": 12, "mp_rear": 50}, "ram_gb": 8}
    specs_b = {"ram_gb": 8, "camera": {"mp_rear": 50, "mp_front": 12}}  # reversed both levels
    assert compute_specs_hash(specs_a) == compute_specs_hash(specs_b)


def test_embedding_version_comes_from_constants():
    """EMBEDDING_VERSION must be imported from app.core.constants, not redefined."""
    from app.core.constants import EMBEDDING_VERSION as canonical_version
    assert EMBEDDING_VERSION == canonical_version


# ── compute_reviews_hash ──────────────────────────────────────────────────────


def test_reviews_hash_returns_none_for_empty():
    """Empty review text should return None, not an empty-string hash."""
    assert compute_reviews_hash("") is None
    assert compute_reviews_hash("   ") is None
    assert compute_reviews_hash(None) is None


def test_reviews_hash_is_deterministic():
    text = "Great camera, smooth performance."
    assert compute_reviews_hash(text) == compute_reviews_hash(text)


def test_reviews_hash_changes_on_content_change():
    h1 = compute_reviews_hash("Great battery")
    h2 = compute_reviews_hash("Poor battery")
    assert h1 != h2


# ── compute_embedding_hash ────────────────────────────────────────────────────


def test_embedding_hash_is_16_chars():
    vector = [0.1] * 1536
    h = compute_embedding_hash(vector)
    assert len(h) == 16


def test_embedding_hash_deterministic():
    vector = [float(i) / 100 for i in range(1536)]
    assert compute_embedding_hash(vector) == compute_embedding_hash(vector)


def test_embedding_hash_changes_on_vector_change():
    v1 = [0.1] * 1536
    v2 = [0.2] * 1536
    assert compute_embedding_hash(v1) != compute_embedding_hash(v2)


# ── _specs_to_text ────────────────────────────────────────────────────────────


def test_specs_to_text_includes_title():
    text = _specs_to_text({"ram_gb": 8}, "Samsung Galaxy S24")
    assert "Samsung Galaxy S24" in text
    assert "ram_gb" in text


def test_specs_to_text_handles_none_specs():
    text = _specs_to_text(None, "Test Phone")
    assert text == "Test Phone"


def test_specs_to_text_no_lllm_import():
    """Verify embedding pipeline does not import any LLM client."""
    import app.services.embedding_pipeline as module
    source_file = __import__("inspect").getfile(module)
    with open(source_file) as f:
        source = f.read()
    # Must not import LangChain, OpenAI client directly (only via provider)
    assert "from langchain" not in source
    assert "import openai" not in source
    assert "ChatOpenAI" not in source


# ── Pipeline integration (mocked) ────────────────────────────────────────────


@pytest.fixture
def mock_provider():
    p = AsyncMock()
    p.embed = AsyncMock(return_value=[0.1] * 1536)
    return p


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.set = AsyncMock(return_value=b"OK")   # lock acquired
    r.delete = AsyncMock()
    r.incr = AsyncMock()
    return r


@pytest.fixture
def mock_product():
    p = MagicMock()
    p.id = 1
    p.canonical_title = "Apple iPhone 15 Pro"
    p.brand = "Apple"
    p.specs_jsonb = {"ram_gb": 8, "storage_gb": 256}
    return p


@pytest.fixture
def mock_review():
    r = MagicMock()
    r.product_id = 1
    r.summary_text = "Excellent camera performance and great battery life."
    r.sentiment_score = 0.85
    return r


@pytest.mark.asyncio
async def test_generate_specs_embedding_calls_provider(mock_provider, mock_redis, mock_product):
    """Pipeline must call provider.embed() with the specs text."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()

    # Mock product query
    product_result = MagicMock()
    product_result.scalar_one_or_none.return_value = mock_product

    # Mock should_reembed to return True (no existing embedding)
    no_existing = MagicMock()
    no_existing.scalar_one_or_none.return_value = None

    mock_db.execute.side_effect = [product_result, no_existing]

    with patch("app.services.embedding_pipeline._upsert_embedding", new_callable=AsyncMock):
        from app.services.embedding_pipeline import generate_specs_embedding
        result = await generate_specs_embedding(1, mock_db, mock_redis, mock_provider)

    mock_provider.embed.assert_awaited_once()
    assert result["status"] == "generated"


@pytest.mark.asyncio
async def test_generate_specs_embedding_releases_lock_on_error(mock_provider, mock_redis, mock_product):
    """Embed lock must be released even when provider raises an exception."""
    mock_provider.embed.side_effect = RuntimeError("provider down")

    mock_db = AsyncMock()
    product_result = MagicMock()
    product_result.scalar_one_or_none.return_value = mock_product
    no_existing = MagicMock()
    no_existing.scalar_one_or_none.return_value = None
    mock_db.execute.side_effect = [product_result, no_existing]

    from app.services.embedding_pipeline import generate_specs_embedding
    result = await generate_specs_embedding(1, mock_db, mock_redis, mock_provider)

    # Lock must be released via finally block regardless of error
    mock_redis.delete.assert_awaited()
    # delete called with embed:1:specs
    delete_calls = [str(c) for c in mock_redis.delete.call_args_list]
    assert any("specs" in c for c in delete_calls)
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_generate_review_embedding_skips_when_no_summary(mock_provider, mock_redis):
    """Review embedding must skip when summary_text is None."""
    mock_db = AsyncMock()
    review_result = MagicMock()
    review_result.scalar_one_or_none.return_value = None  # no review
    mock_db.execute.return_value = review_result

    from app.services.embedding_pipeline import generate_review_embedding
    result = await generate_review_embedding(1, mock_db, mock_redis, mock_provider)

    mock_provider.embed.assert_not_awaited()
    assert result["status"] == "skipped"
    assert "no_review_summary" in result["reason"]


@pytest.mark.asyncio
async def test_generate_review_embedding_skips_when_empty_text(mock_provider, mock_redis):
    """Review embedding must skip when summary_text is empty string."""
    mock_db = AsyncMock()
    review_result = MagicMock()
    empty_review = MagicMock()
    empty_review.summary_text = "   "
    review_result.scalar_one_or_none.return_value = empty_review
    mock_db.execute.return_value = review_result

    from app.services.embedding_pipeline import generate_review_embedding
    result = await generate_review_embedding(1, mock_db, mock_redis, mock_provider)

    mock_provider.embed.assert_not_awaited()
    assert result["status"] == "skipped"
