"""POST /v1/chat — Primary unified AI product decision endpoint.

Handles all user intents:
    - SEARCH_PRODUCT
    - COMPARE_PRODUCTS
    - FEATURE_QUERY
    - RECOMMEND_PRODUCTS
    - PRICE_QUERY

All queries flow through the same LangGraph pipeline.
Session memory via session_id enables unlimited follow-up questions.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends

from app.api.deps import get_embedding_provider, get_llm_client, get_redis
from app.graph import compile_graph
from app.models.database import async_session_factory
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    redis: object = Depends(get_redis),
    llm_client: object = Depends(get_llm_client),
    embedding_provider: object = Depends(get_embedding_provider),
) -> ChatResponse:
    """AI Product Decision Engine — unified endpoint.

    Pass session_id from a previous response to enable follow-up questions.
    The engine remembers products, preferences, and conversation context.

    Returns a structured_output payload with UI-renderable components:
        - recommendation_cards: ranked product recommendations
        - comparison_table: head-to-head product comparison
        - product_detail: deep technical feature breakdown
        - review_summary: real-world user feedback

    All outputs are schema-versioned (version: "v1") for frontend stability.
    """
    session_id = request.session_id or str(uuid4())

    graph = compile_graph(
        redis_client=redis,
        db_session=async_session_factory,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
    )

    result = await graph.ainvoke({
        "query": request.query,
        "session_id": session_id,
    })

    response_data = result.get("response") or {}
    structured_output = response_data.get("structured_output") or result.get("structured_output") or {}
    source = response_data.get("source", "ai")
    trace_id = result.get("trace_id") or response_data.get("trace_id") or ""
    out_session_id = response_data.get("session_id") or result.get("session_id") or session_id

    logger.info(
        "chat_response_sent",
        session_id=out_session_id,
        intent_type=structured_output.get("intent_type", "unknown"),
        component_count=len(structured_output.get("components", [])),
        source=source,
    )

    return ChatResponse(
        session_id=out_session_id,
        structured_output=structured_output,
        source=source,
        trace_id=trace_id,
    )
