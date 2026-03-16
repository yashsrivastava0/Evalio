"""Chat endpoint request/response schemas.

POST /v1/chat is the primary unified endpoint for all user intents.
All queries — search, compare, feature, recommend — flow through here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the /v1/chat endpoint."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Natural language product query",
    )
    session_id: str | None = Field(
        None,
        description="Session ID for follow-up questions. "
                    "Pass null on first query; reuse on follow-ups.",
    )


class ChatResponse(BaseModel):
    """Response from the /v1/chat endpoint."""

    session_id: str = Field(..., description="Session ID — reuse for follow-up questions")
    structured_output: dict[str, Any] = Field(
        ...,
        description="StructuredOutput payload (version, components, intent_type, etc.)",
    )
    source: str = Field(
        "ai",
        description="'ai' for fresh response, 'cache' for cached response",
    )
    trace_id: str = Field(
        "",
        description="Trace ID for debugging — use GET /v1/trace/{trace_id}",
    )
