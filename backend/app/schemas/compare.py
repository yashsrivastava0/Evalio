"""Pydantic schemas for /v1/compare endpoint.

Comparison of 2–4 products with precomputed numeric diffs.
Explanation Agent renders prose — it cannot reorder products.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CompareRequest(BaseModel):
    """Request to compare 2–4 products."""

    product_slugs: list[str] = Field(
        ..., min_length=2, max_length=4, description="Product slugs to compare"
    )
    aspects: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Aspects to focus on (e.g., camera, battery)",
    )
    session_id: str | None = Field(None, description="Session ID")

    @field_validator("product_slugs", mode="before")
    @classmethod
    def lowercase_slugs(cls, v: list[str]) -> list[str]:
        return [s.lower().strip() for s in v]


class NumericDiff(BaseModel):
    """Precomputed numeric difference between two products for one aspect."""

    aspect: str
    product_a_slug: str
    product_a_value: float
    product_b_slug: str
    product_b_value: float
    diff: float  # a - b
    winner: str  # slug of the product with higher score


class ComparisonTableRow(BaseModel):
    """One row in the comparison table."""

    aspect: str
    values: dict[str, float]  # slug → score
    winner: str


class ProCon(BaseModel):
    """Pros and cons for a single product."""

    product_slug: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    """Complete comparison response.

    ranked_list order comes from deterministic RankingEngine.
    ExplanationAgent cannot reorder this list.
    """

    ranked_list: list[str] = Field(
        description="Product slugs ordered by deterministic ranking"
    )
    comparison_table: list[ComparisonTableRow] = Field(default_factory=list)
    numeric_diffs: list[NumericDiff] = Field(default_factory=list)
    pros_cons: list[ProCon] = Field(default_factory=list)
    verdict: str = Field("", description="LLM-generated verdict (prose only, no arithmetic)")
    meta: dict[str, Any] = Field(default_factory=dict)
