"""Query Expansion Agent — I/O schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExpandedQuery(BaseModel):
    """Output schema for the Query Expansion Agent."""

    original: str = Field(..., description="The original user query")
    search_criteria: list[str] = Field(
        default_factory=list,
        description="Concrete technical terms that capture the user's need. "
                    "E.g. ['optical image stabilization', '4K 60fps', 'large sensor']",
    )
    feature_synonyms: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping of vague terms to technical synonyms. "
                    "E.g. {'vlogging': ['stabilization', 'front camera', 'HEVC']}",
    )
    sql_filters: dict = Field(
        default_factory=dict,
        description="Structured filter parameters: category, budget_min, budget_max, "
                    "brands (list), min_ram, min_storage",
    )
    vector_query_text: str = Field(
        ...,
        description="Rewritten query optimized for semantic vector search",
    )
    attribute_priorities: dict[str, float] = Field(
        default_factory=dict,
        description="Normalized weights for key scoring dimensions. "
                    "E.g. {'camera': 0.5, 'battery': 0.3, 'performance': 0.2}. Sum ≈ 1.0. "
                    "Empty dict means equal weight across all attributes.",
    )
    intent_type: str = Field(
        ...,
        description="Intent type passed through from Intent Agent",
    )
