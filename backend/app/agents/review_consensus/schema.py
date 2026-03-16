"""Review Consensus Agent — I/O schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductReviewSummary(BaseModel):
    """Structured review summary for a single product."""

    product_slug: str
    sentiment_score: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Sentiment score: 0=negative, 1=positive",
    )
    sentiment_label: str = Field(
        "unknown",
        description="'positive', 'mixed', 'negative', or 'unknown'",
    )
    summary_text: str = ""
    source_count: int = 0
    last_synced: str | None = None
    key_praise: list[str] = Field(
        default_factory=list,
        description="2-3 specific things real users praised",
    )
    key_complaints: list[str] = Field(
        default_factory=list,
        description="1-2 specific things real users complained about",
    )


class ReviewConsensusData(BaseModel):
    """Map of product_slug → review summary."""

    summaries: dict[str, ProductReviewSummary] = Field(default_factory=dict)


class ReviewConsensus(BaseModel):
    """Output schema for the Review Consensus Agent."""

    component: Literal["review_summary"] = "review_summary"
    ui_hint: Literal["render as list"] = "render as list"
    data: ReviewConsensusData
