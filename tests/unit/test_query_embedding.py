"""Unit tests for query embedding node.

Verifies:
    1. Query normalization (lowercase, strip punctuation, collapse whitespace)
    2. Deterministic hashing
    3. Filter hash prevents budget-range collisions
    4. Embedding hash from vector
"""

from __future__ import annotations

import pytest

from app.agents.query_embedding import (
    normalize_query,
    compute_query_hash,
    compute_filter_hash,
    compute_embedding_hash,
)
from app.agents.router import IntentPlan


# ── Normalization ──

def test_normalize_lowercase():
    assert normalize_query("BEST Phone") == "best phone"


def test_normalize_strip_punctuation():
    assert normalize_query("best phone!!! under ₹30,000?") == "best phone under 30000"


def test_normalize_collapse_whitespace():
    assert normalize_query("best   phone    under  30000") == "best phone under 30000"


def test_normalize_idempotent():
    q = "best phone under 30000"
    assert normalize_query(q) == normalize_query(normalize_query(q))


# ── Query Hash ──

def test_hash_deterministic():
    q = "best camera phone"
    assert compute_query_hash(q) == compute_query_hash(q)


def test_hash_different_queries():
    assert compute_query_hash("best phone") != compute_query_hash("best laptop")


# ── Filter Hash ──

def test_filter_hash_prevents_budget_collision():
    """Same query, different budgets → different filter hashes."""
    intent_30k = IntentPlan(category="smartphone", budget_max=30000)
    intent_40k = IntentPlan(category="smartphone", budget_max=40000)

    hash_30k = compute_filter_hash(intent_30k)
    hash_40k = compute_filter_hash(intent_40k)

    assert hash_30k != hash_40k


def test_filter_hash_same_intent():
    """Same intent → same filter hash."""
    intent1 = IntentPlan(category="smartphone", budget_max=30000, brands=["samsung"])
    intent2 = IntentPlan(category="smartphone", budget_max=30000, brands=["samsung"])

    assert compute_filter_hash(intent1) == compute_filter_hash(intent2)


def test_filter_hash_brand_order_independent():
    """Brand order shouldn't matter (sorted internally)."""
    intent1 = IntentPlan(brands=["samsung", "apple"])
    intent2 = IntentPlan(brands=["apple", "samsung"])

    assert compute_filter_hash(intent1) == compute_filter_hash(intent2)


# ── Embedding Hash ──

def test_embedding_hash_deterministic():
    emb = [0.1, 0.2, 0.3] * 100
    assert compute_embedding_hash(emb) == compute_embedding_hash(emb)


def test_embedding_hash_different_vectors():
    emb1 = [0.1] * 100
    emb2 = [0.2] * 100
    assert compute_embedding_hash(emb1) != compute_embedding_hash(emb2)
