"""Intent Understanding Agent — I/O schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """Output schema for the Intent Understanding Agent.

    The LLM must return JSON matching this schema exactly.
    """

    intent_type: Literal[
        "SEARCH_PRODUCT",
        "COMPARE_PRODUCTS",
        "FEATURE_QUERY",
        "RECOMMEND_PRODUCTS",
        "PRICE_QUERY",
        "PRODUCT_DEEP_DIVE",       # full technical profile of one specific product
        "TECHNICAL_EXPLANATION",   # explain a feature/concept in context of a product
        "FOLLOWUP_QUERY",          # explicit follow-up referencing session context
        "INVALID_QUERY",           # off-topic, harmful, or unrecognisable queries
    ] = Field(..., description="Classified user intent")

    entities: list[str] = Field(
        default_factory=list,
        description="Extracted product names or model identifiers",
    )
    budget_min: float | None = Field(None, description="Minimum budget in INR")
    budget_max: float | None = Field(None, description="Maximum budget in INR")
    brands: list[str] = Field(
        default_factory=list,
        description="Brand preferences extracted from query",
    )
    category: str | None = Field(
        None,
        description="Product category: 'smartphone' or 'laptop'",
    )
    focus_attributes: list[str] = Field(
        default_factory=list,
        description="User's priority attributes e.g. ['camera', 'battery', 'gaming']",
    )
    comparison_targets: list[str] = Field(
        default_factory=list,
        description="Specific product identifiers to compare (for COMPARE_PRODUCTS)",
    )
    is_followup: bool = Field(
        False,
        description="True if this query references a previous result in the session",
    )
    followup_reference: str | None = Field(
        None,
        description="What the user is referring to from session context",
    )
    raw_query: str = Field(..., description="The original user query unchanged")


class IntentAgentInput(BaseModel):
    """Input passed to the intent agent's prompt template."""

    query: str
    session_history: list[dict] = Field(default_factory=list)
    session_products: list[str] = Field(default_factory=list)
    last_query: str = ""
    last_intent_type: str = ""
