"""Shared hashing and normalization utilities for all agents.

Moved from agents/query_embedding.py to avoid circular imports and allow
reuse across the agent layer.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


def normalize_query(query: str) -> str:
    """Normalize query for consistent hashing.

    - lowercase
    - strip punctuation
    - collapse whitespace
    """
    q = query.lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def compute_query_hash(normalized_query: str) -> str:
    """Compute deterministic hash of normalized query string."""
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]


def compute_embedding_hash(embedding: list[float]) -> str:
    """Compute hash of embedding vector for semantic cache key."""
    key_floats = embedding[:32] + embedding[-32:] if len(embedding) > 64 else embedding
    raw = "|".join(f"{f:.6f}" for f in key_floats)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_filter_hash(filters: dict[str, Any]) -> str:
    """Compute hash of SQL filter parameters to prevent cache collisions.

    Accepts a plain dict (from ExpandedQuery.sql_filters) so agents don't
    need to import the old IntentPlan dataclass.
    """
    parts = [
        str(filters.get("category") or ""),
        str(filters.get("budget_min") or ""),
        str(filters.get("budget_max") or ""),
        ",".join(sorted(filters.get("brands") or [])),
        str(filters.get("min_ram") or ""),
        str(filters.get("min_storage") or ""),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_intent_filter_hash(intent: Any) -> str:
    """Compute filter hash directly from an intent object.

    Compatibility shim — accepts old-style intent dataclasses/dicts.
    """
    if isinstance(intent, dict):
        return compute_filter_hash(intent)
    return compute_filter_hash({
        "category": getattr(intent, "category", None),
        "budget_min": getattr(intent, "budget_min", None),
        "budget_max": getattr(intent, "budget_max", None),
        "brands": getattr(intent, "brands", []),
        "min_ram": getattr(intent, "min_ram", None),
        "min_storage": getattr(intent, "min_storage", None),
    })
