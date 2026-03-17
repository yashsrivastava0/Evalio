"""Comparison Reasoning Agent — I/O schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ComparedProduct(BaseModel):
    """A single product in the comparison output."""

    product_id: int
    product_slug: str
    canonical_title: str
    price_range: str = "Price unavailable"
    pros: list[str] = Field(default_factory=list, description="Key strengths for this use case")
    cons: list[str] = Field(default_factory=list, description="Key weaknesses")


class TechnicalDiff(BaseModel):
    """A single technical difference between compared products."""

    attribute: str = Field(..., description="What is being compared, e.g. 'Camera'")
    values: dict[str, Any] = Field(
        ...,
        description="product_slug → value string, e.g. {'samsung-s24': '200MP main sensor'}",
    )
    winner: str | None = Field(None, description="product_slug of the winner, or null if tied")
    explanation: str = Field(..., description="One sentence explaining why the winner wins")


class ComparisonData(BaseModel):
    """Structured comparison data."""

    products: list[ComparedProduct]
    technical_diffs: list[TechnicalDiff]
    use_case_suitability: dict[str, list[str]] = Field(
        default_factory=dict,
        description="product_slug → list of use cases it excels at",
    )
    per_category_winners: dict[str, str | None] = Field(
        default_factory=dict,
        description="Category → winning product_slug, or null if tied. "
                    "Keys: camera, performance, battery, display, value, build",
    )
    overall_recommendation: str = Field(
        ...,
        description="One sentence overall recommendation for a typical buyer",
    )
    tradeoffs: list[str] = Field(
        default_factory=list,
        description="Key tradeoffs the buyer must accept",
    )
    real_world_implications: list[str] = Field(
        default_factory=list,
        description="What the technical differences mean in daily use. "
                    "E.g. 'S24 LTPO display saves ~15% battery during light use'",
    )


class ComparisonResult(BaseModel):
    """Output schema for the Comparison Reasoning Agent."""

    component: Literal["comparison_table"] = "comparison_table"
    ui_hint: Literal["render as table"] = "render as table"
    data: ComparisonData
