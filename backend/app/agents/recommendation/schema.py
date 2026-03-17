"""Recommendation Agent — I/O schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MatchScoreBreakdown(BaseModel):
    """Per-attribute match scores for UI visualization (0–10 scale)."""

    camera: int | None = None
    gaming: int | None = None
    battery: int | None = None
    performance: int | None = None
    display: int | None = None
    value_for_money: int | None = None
    build_quality: int | None = None
    portability: int | None = None


class RecommendedProduct(BaseModel):
    """A single recommended product with match scores and reasons."""

    product_id: int
    product_slug: str
    canonical_title: str
    match_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Overall match score from 0.0 to 1.0",
    )
    match_score_breakdown: MatchScoreBreakdown = Field(
        default_factory=MatchScoreBreakdown,
        description="Per-category scores for UI bar/radar chart",
    )
    reason: str = Field(
        ...,
        description="One concise sentence: why this product fits the user's need",
    )
    pros: list[str] = Field(
        default_factory=list,
        description="2-3 specific strengths for this use case",
    )
    cons: list[str] = Field(
        default_factory=list,
        description="1-2 notable weaknesses for this use case",
    )
    best_use_cases: list[str] = Field(
        default_factory=list,
        description="Top 2-3 use cases this product excels at. E.g. ['vlogging', 'travel photography']",
    )
    not_suitable_for: list[str] = Field(
        default_factory=list,
        description="1-2 use cases where this product underperforms. E.g. ['heavy gaming', '4K video editing']",
    )
    price: float | None = Field(None, description="Best available price in INR")


class RecommendationData(BaseModel):
    """Structured recommendation data."""

    ranked: list[RecommendedProduct]
    rationale: str = Field(
        ...,
        description="1-2 sentences explaining the overall ranking logic",
    )
    user_intent_summary: str = Field(
        ...,
        description="One sentence rephrasing what the user needs",
    )


class RecommendationResult(BaseModel):
    """Output schema for the Recommendation Agent."""

    component: Literal["recommendation_cards"] = "recommendation_cards"
    ui_hint: Literal["render as cards"] = "render as cards"
    data: RecommendationData
