"""Unit tests — embedding dedupe logic (should_reembed).

Tests:
    - Returns False when specs_hash unchanged and version matches
    - Returns True when specs_hash changed
    - Returns True when embedding_version bumped
    - Pipeline skips provider.embed() when should_reembed returns False
    - Logs "dedupe_skip" on skip
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embedding_pipeline import should_reembed, EMBEDDING_VERSION


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_existing(
    specs_hash: str = "aabbcc",
    reviews_hash: str | None = None,
    version: int = EMBEDDING_VERSION,
) -> MagicMock:
    """Create a mock existing Embedding row."""
    emb = MagicMock()
    emb.specs_hash = specs_hash
    emb.reviews_hash = reviews_hash
    emb.embedding_version = version
    return emb


# ── should_reembed ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_reembed_when_hash_and_version_match():
    """Skip when stored hash == new hash AND version is current."""
    mock_db = AsyncMock()
    existing = _make_existing(specs_hash="abc123", version=EMBEDDING_VERSION)
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing
    mock_db.execute.return_value = result_mock

    result = await should_reembed(1, mock_db, "specs", "abc123")
    assert result is False


@pytest.mark.asyncio
async def test_reembed_when_specs_hash_changed():
    """Force re-embed when stored hash != new hash."""
    mock_db = AsyncMock()
    existing = _make_existing(specs_hash="oldHash", version=EMBEDDING_VERSION)
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing
    mock_db.execute.return_value = result_mock

    result = await should_reembed(1, mock_db, "specs", "newHash")
    assert result is True


@pytest.mark.asyncio
async def test_reembed_when_version_bumped():
    """Force re-embed when stored version < current EMBEDDING_VERSION."""
    mock_db = AsyncMock()
    # Simulate stored version is behind by 1
    existing = _make_existing(specs_hash="sameHash", version=EMBEDDING_VERSION - 1)
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = existing
    mock_db.execute.return_value = result_mock

    result = await should_reembed(1, mock_db, "specs", "sameHash")
    assert result is True


@pytest.mark.asyncio
async def test_reembed_when_no_existing_embedding():
    """Always generate when no embedding exists for this product/type."""
    mock_db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None  # no existing row
    mock_db.execute.return_value = result_mock

    result = await should_reembed(1, mock_db, "specs", "anyHash")
    assert result is True


@pytest.mark.asyncio
async def test_no_reembed_when_content_hash_none():
    """Returns False when new_content_hash is None (no data guard)."""
    mock_db = AsyncMock()
    result = await should_reembed(1, mock_db, "review", None)
    assert result is False
    # DB should not even be queried
    mock_db.execute.assert_not_awaited()


# ── Pipeline respects should_reembed ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_skips_embed_when_hash_unchanged():
    """Pipeline must not call provider.embed() when dedupe returns False."""
    mock_provider = AsyncMock()
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock()

    mock_db = AsyncMock()

    # Product query
    product = MagicMock()
    product.id = 1
    product.canonical_title = "Test Phone"
    product.specs_jsonb = {"ram_gb": 8}

    product_result = MagicMock()
    product_result.scalar_one_or_none.return_value = product

    # Existing embedding with matching hash
    from app.services.embedding_pipeline import compute_specs_hash
    existing_hash = compute_specs_hash({"ram_gb": 8})
    existing = _make_existing(specs_hash=existing_hash, version=EMBEDDING_VERSION)
    existing_result = MagicMock()
    existing_result.scalar_one_or_none.return_value = existing

    mock_db.execute.side_effect = [product_result, existing_result]

    from app.services.embedding_pipeline import generate_specs_embedding
    result = await generate_specs_embedding(1, mock_db, mock_redis, mock_provider)

    mock_provider.embed.assert_not_awaited()
    assert result["status"] == "skipped"
    assert "dedupe" in result["reason"]


@pytest.mark.asyncio
async def test_pipeline_increments_skip_counter_on_dedupe():
    """Skip counter must increment when pipeline dedupes."""
    mock_provider = AsyncMock()
    mock_redis = AsyncMock()
    mock_redis.incr = AsyncMock()

    mock_db = AsyncMock()

    product = MagicMock()
    product.id = 2
    product.canonical_title = "Test Phone"
    product.specs_jsonb = {"storage_gb": 64}

    product_result = MagicMock()
    product_result.scalar_one_or_none.return_value = product

    from app.services.embedding_pipeline import compute_specs_hash
    existing_hash = compute_specs_hash({"storage_gb": 64})
    existing = _make_existing(specs_hash=existing_hash, version=EMBEDDING_VERSION)
    existing_result = MagicMock()
    existing_result.scalar_one_or_none.return_value = existing

    mock_db.execute.side_effect = [product_result, existing_result]

    from app.services.embedding_pipeline import generate_specs_embedding
    with patch("app.services.embedding_pipeline._track_skip", new_callable=AsyncMock) as track_skip:
        await generate_specs_embedding(2, mock_db, mock_redis, mock_provider)
        track_skip.assert_awaited_once()
