"""Pydantic schemas for /v1/search endpoint.

Strict types, validated enums, and structured response with timing metadata.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CategoryFilter(str, Enum):
    """Product category — MVP: smartphones and laptops only."""

    smartphone = "smartphone"
    laptop = "laptop"


class SortBy(str, Enum):
    """Sort options for search results."""

    relevance = "relevance"
    price_asc = "price_asc"
    price_desc = "price_desc"
    rating = "rating"
    newest = "newest"


class SearchRequest(BaseModel):
    """Search request schema."""

    query: str = Field(..., min_length=2, max_length=500, description="Natural language search query")
    category: CategoryFilter | None = Field(None, description="Filter by category")
    budget_min: float | None = Field(None, ge=0, description="Minimum budget (INR)")
    budget_max: float | None = Field(None, ge=0, description="Maximum budget (INR)")
    brands: list[str] = Field(default_factory=list, max_length=10, description="Brand filter")
    min_ram: int | None = Field(None, ge=1, le=128, description="Minimum RAM in GB")
    min_storage: int | None = Field(None, ge=1, le=8192, description="Minimum storage in GB")
    sort_by: SortBy = Field(SortBy.relevance, description="Sort order")
    page: int = Field(1, ge=1, le=100, description="Page number")
    limit: int = Field(10, ge=1, le=50, description="Results per page")
    session_id: str | None = Field(None, description="Session ID for continuity")

    @field_validator("brands", mode="before")
    @classmethod
    def lowercase_brands(cls, v: list[str]) -> list[str]:
        return [b.lower().strip() for b in v] if v else []


class ProductResult(BaseModel):
    """Single product in search results."""

    product_id: int
    product_slug: str
    brand: str
    model_name: str
    canonical_title: str
    category: str
    price_inr: float | None = None
    availability: bool = True
    ram_gb: int | None = None
    storage_gb: int | None = None
    vendor: str | None = None
    affiliate_url: str | None = None
    specs: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    """Deterministic scoring breakdown for a product."""

    final_score: float
    performance_score: float = 0.0
    battery_score: float = 0.0
    camera_score: float = 0.0
    display_score: float = 0.0
    build_score: float = 0.0
    price_score: float = 0.0
    popularity_score: float = 0.0


class RankedProduct(BaseModel):
    """Product with its ranking score."""

    rank: int
    product: ProductResult
    scores: ScoreBreakdown
    explanation_snippet: str = ""


class SearchMeta(BaseModel):
    """Metadata about the search response."""

    total_candidates: int = 0
    total_filtered: int = 0
    total_ranked: int = 0
    page: int = 1
    limit: int = 10
    cache_hit: bool = False
    cache_type: str | None = None  # "exact_query" | "semantic" | None
    timing_ms: dict[str, float] = Field(default_factory=dict)
    route: str = ""  # "cache_hit" | "fast_path" | "fetch_trigger"


class SearchResponse(BaseModel):
    """Complete search response."""

    results: list[RankedProduct] = Field(default_factory=list)
    meta: SearchMeta = Field(default_factory=SearchMeta)
    explanation: dict[str, Any] = Field(default_factory=dict)
