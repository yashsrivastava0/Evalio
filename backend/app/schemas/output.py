"""Structured Output schemas — final payload sent to the API layer and UI.

All agent outputs are assembled into a StructuredOutput by the output_generator.
The frontend renders components from the components list.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OutputComponent(BaseModel):
    """A single UI-renderable component produced by an agent."""

    component: str = Field(
        ...,
        description="Component type: 'comparison_table' | 'recommendation_cards' | "
                    "'product_detail' | 'review_summary'",
    )
    ui_hint: str = Field(
        ...,
        description="Render hint for the frontend: 'render as table' | 'render as cards' | etc.",
    )
    data: dict[str, Any] = Field(..., description="Agent-specific payload")


class StructuredOutput(BaseModel):
    """Final versioned output payload returned to the API layer.

    The frontend depends on this contract. Breaking changes must bump version.
    """

    version: str = Field("v1", description="Schema version — bump on breaking changes")
    session_id: str
    trace_id: str
    intent_type: str = Field(
        ...,
        description="Classified intent: SEARCH_PRODUCT | COMPARE_PRODUCTS | etc.",
    )
    components: list[OutputComponent] = Field(
        default_factory=list,
        description="Ordered list of UI components to render",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="confidence_score (0.0-1.0), sources (list[str]), retrieval_method (str)",
    )


class PartialResponse(BaseModel):
    """Partial response returned when data is being fetched in the background."""

    version: str = "v1"
    session_id: str
    trace_id: str
    status: str = "fetching"
    message: str = "Product data is being fetched. Please retry in a few seconds."
    retry_after_seconds: int = 5
