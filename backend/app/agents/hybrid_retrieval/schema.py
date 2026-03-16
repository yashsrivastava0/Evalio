"""Hybrid Retrieval Agent — I/O schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductCandidate(BaseModel):
    """A single candidate product returned by the retrieval pipeline."""

    product_id: int
    product_slug: str
    canonical_title: str
    brand: str
    category: str
    specs_jsonb: dict = Field(default_factory=dict)
    variants: list[dict] = Field(default_factory=list)
    vector_distance: float = 1.0
    scores: dict = Field(
        default_factory=dict,
        description="Component scores from DB (performance, battery, camera, etc.)",
    )
    retrieval_score: float = Field(
        0.0,
        description="Composite reranking score: 0.4×similarity + 0.3×price + 0.2×brand + 0.1×recency",
    )


class HybridRetrievalOutput(BaseModel):
    """Output schema for the Hybrid Retrieval Agent."""

    candidates: list[ProductCandidate] = Field(default_factory=list)
    retrieval_route: Literal[
        "cache_hit",
        "session_bypass",
        "sql_vector",
        "entity_resolved",
        "fetch_trigger",
    ] = "sql_vector"
    cache_payload: dict | None = None
    fetch_needed: bool = False
    total_sql_candidates: int = 0
    total_vector_candidates: int = 0
