"""Evalio AI Product Decision Engine — LangGraph Root Graph.

New 8-node architecture replacing the old deterministic pipeline.

Flow:
    session_context → intent_understanding → query_expansion → hybrid_retrieval
                                                                     ↓
                                           ┌─────────────────────────┼────────────────────┐
                                      cache_hit              fetch_trigger          sql_vector
                                           ↓                        ↓                     ↓
                                        response           fetch_trigger          review_consensus
                                                                  ↓               ↙          ↘
                                                               response      comparison   recommendation
                                                                             reasoning          ↓
                                                                                 ↓      structured_output
                                                                         structured_output    ↓
                                                                                 ↓      session_update
                                                                          session_update       ↓
                                                                                 ↓          response
                                                                              response

    feature_query branch: hybrid_retrieval → product_knowledge → structured_output → ...

Core principles:
    - AI should reason, code should retrieve data. Never mix.
    - All agents return structured JSON (never raw text).
    - Session memory enables unlimited follow-up questions.
    - Error recovery: each LLM node retries 2× before fallback.
"""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, END

from app.agents.session_context import SessionData, save_session, load_session
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── GraphState ────────────────────────────────────────────────────────────────


class GraphState(TypedDict, total=False):
    """Complete state flowing through the LangGraph.

    All nodes read from and write to this dict.
    No other state mutation mechanism is allowed.
    """

    # Input
    query: str
    session_id: str

    # Node 1: SessionContext
    session: SessionData
    trace_id: str

    # Node 2: IntentUnderstanding
    intent: dict[str, Any]          # IntentResult as dict

    # Node 3: QueryExpansion
    expanded_query: dict[str, Any]  # ExpandedQuery as dict
    query_embedding: list[float]

    # Node 4: HybridRetrieval
    candidates: list[dict[str, Any]]
    retrieval_route: str            # "cache_hit"|"session_bypass"|"sql_vector"|"entity_resolved"|"fetch_trigger"
    cache_payload: dict[str, Any] | None
    fetch_needed: bool

    # Node 5a: ComparisonReasoning
    comparison_result: dict[str, Any]

    # Node 5b: Recommendation
    recommendation_result: dict[str, Any]

    # Node 5c: ProductKnowledge
    product_knowledge_result: dict[str, Any]

    # Node 6: ReviewConsensus (runs before branching to comparison/recommendation)
    review_consensus: dict[str, Any]

    # Node 7: StructuredOutput
    validated_output: dict[str, Any]  # after response_validator
    structured_output: dict[str, Any]  # final UI payload

    # Response
    response: dict[str, Any]


# ── Node factories ────────────────────────────────────────────────────────────


def _make_session_context_node(redis_client: Any):
    """Node 1: Load session from Redis, assign trace_id."""
    async def node(state: GraphState) -> dict:
        session_id = state.get("session_id", "") or str(uuid4())
        session = await load_session(redis_client, session_id)
        trace_id = str(uuid4())
        logger.info("session_loaded", session_id=session_id, turn=session.turn_count, trace_id=trace_id[:8])
        return {
            "session_id": session_id,
            "session": session,
            "trace_id": trace_id,
        }
    return node


def _make_intent_node(llm_client: Any):
    """Node 2: Intent Understanding Agent."""
    from app.agents.intent_understanding.agent import intent_understanding_node
    async def node(state: GraphState) -> dict:
        return await intent_understanding_node(state, llm_client=llm_client)
    return node


def _make_query_expansion_node(llm_client: Any, embedding_provider: Any):
    """Node 3: Query Expansion Agent."""
    from app.agents.query_expansion.agent import query_expansion_node
    async def node(state: GraphState) -> dict:
        return await query_expansion_node(
            state, llm_client=llm_client, embedding_provider=embedding_provider
        )
    return node


def _make_hybrid_retrieval_node(db_session: Any, redis_client: Any):
    """Node 4: Hybrid Retrieval Agent."""
    from app.agents.hybrid_retrieval.agent import hybrid_retrieval_node
    async def node(state: GraphState) -> dict:
        async with db_session() as session:
            return await hybrid_retrieval_node(
                state, db_session=session, redis_client=redis_client
            )
    return node


def _make_review_consensus_node(db_session: Any, llm_client: Any):
    """Node 5: Review Consensus Agent."""
    from app.agents.review_consensus.agent import review_consensus_node
    async def node(state: GraphState) -> dict:
        async with db_session() as session:
            return await review_consensus_node(state, db_session=session, llm_client=llm_client)
    return node


def _make_comparison_reasoning_node(llm_client: Any):
    """Node 6a: Comparison Reasoning Agent."""
    from app.agents.comparison_reasoning.agent import comparison_reasoning_node
    async def node(state: GraphState) -> dict:
        return await comparison_reasoning_node(state, llm_client=llm_client)
    return node


def _make_recommendation_node(llm_client: Any):
    """Node 6b: Recommendation Agent."""
    from app.agents.recommendation.agent import recommendation_node
    async def node(state: GraphState) -> dict:
        return await recommendation_node(state, llm_client=llm_client)
    return node


def _make_product_knowledge_node(db_session: Any, llm_client: Any):
    """Node 6c: Product Knowledge Agent."""
    from app.agents.product_knowledge.agent import product_knowledge_node
    async def node(state: GraphState) -> dict:
        async with db_session() as session:
            return await product_knowledge_node(state, db_session=session, llm_client=llm_client)
    return node


def _make_fetch_trigger_node(redis_client: Any, celery_available: bool = True):
    """Node: Fire Celery scrape task (non-blocking). Return partial response."""
    async def node(state: GraphState) -> dict:
        intent = state.get("intent") or {}
        query = state.get("query", "")

        # Extract brand/model hints from query for targeted scraping
        entities = intent.get("entities", []) if isinstance(intent, dict) else []
        brands = intent.get("brands", []) if isinstance(intent, dict) else []
        slug_hint = (entities[0] if entities else brands[0] if brands else query[:30]).lower().replace(" ", "-")

        # Always attempt dispatch — task queues in Redis broker and is processed
        # whenever a Celery worker is running (fire-and-forget, fail-silent).
        try:
            from app.workers.tasks import celery_app
            celery_app.send_task("tasks.scrape_product", args=[slug_hint])
            logger.info("fetch_trigger_fired", slug_hint=slug_hint)
        except Exception as e:
            logger.info("fetch_trigger_skipped", slug_hint=slug_hint, error=str(e))

        from app.schemas.output import PartialResponse
        partial = PartialResponse(
            session_id=state.get("session_id", ""),
            trace_id=state.get("trace_id", ""),
        )
        return {"response": partial.model_dump()}
    return node


def _make_structured_output_node(db_session: Any):
    """Node 7: Assemble structured output from agent results."""
    from app.services.output_generator import structured_output_node
    from app.services.response_validator import verify_agent_output
    async def node(state: GraphState) -> dict:
        raw_output = await structured_output_node(state)
        async with db_session() as session:
            validated = await verify_agent_output(raw_output.get("structured_output", {}), session)
        return {"structured_output": validated, "validated_output": validated}
    return node


def _make_session_update_node(redis_client: Any):
    """Node 8: Persist updated session to Redis."""
    async def node(state: GraphState) -> dict:
        session = state.get("session")
        if not session:
            return {}

        candidates = state.get("candidates", [])
        intent = state.get("intent") or {}

        # Update last_results
        session.last_results = candidates[:10]

        # Update active_comparison
        comparison_result = state.get("comparison_result") or {}
        if comparison_result:
            compared = comparison_result.get("data", {}).get("products", [])
            session.active_comparison = [p.get("product_slug", "") for p in compared[:4] if p.get("product_slug")]

        # Append to conversation history (FIFO cap at 5)
        top_slugs = [c.get("product_slug", "") for c in candidates[:3]]
        session.session_products = list(
            dict.fromkeys(session.session_products + top_slugs)
        )[:20]  # cap at 20 unique products seen

        entry = {
            "query": state.get("query", ""),
            "intent_type": intent.get("intent_type", "") if isinstance(intent, dict) else "",
            "top_products": top_slugs,
        }
        session.conversation_history.append(entry)
        if len(session.conversation_history) > 5:
            session.conversation_history = session.conversation_history[-5:]

        # Update preferences
        if isinstance(intent, dict):
            focus = intent.get("focus_attributes", [])
            budget = intent.get("budget_max")
            if focus:
                session.user_preferences["focus"] = focus
            if budget:
                session.user_preferences["budget_max"] = budget
            session.last_intent_type = intent.get("intent_type", "")
        session.last_query = state.get("query", "")
        session.turn_count += 1

        await save_session(redis_client, session)
        logger.info("session_updated", session_id=session.session_id, turn=session.turn_count)
        return {}
    return node


def _make_response_node():
    """Final node: assemble the HTTP response dict."""
    async def node(state: GraphState) -> dict:
        # Cache hit path
        if state.get("cache_payload"):
            return {"response": {
                **state["cache_payload"],
                "source": "cache",
                "trace_id": state.get("trace_id", ""),
                "session_id": state.get("session_id", ""),
            }}

        # Fetch trigger path (partial response already set)
        if state.get("response"):
            return {}

        # Normal path
        intent = state.get("intent") or {}
        structured = state.get("structured_output") or {}

        response = {
            "session_id": state.get("session_id", ""),
            "structured_output": structured,
            "source": "ai",
            "trace_id": state.get("trace_id", ""),
        }
        return {"response": response}
    return node


# ── Routing functions ─────────────────────────────────────────────────────────


def _route_after_retrieval(state: GraphState) -> str:
    """Branch after hybrid_retrieval based on route + intent."""
    retrieval_route = state.get("retrieval_route", "sql_vector")

    if retrieval_route == "cache_hit":
        return "response"

    if retrieval_route == "fetch_trigger" or state.get("fetch_needed"):
        return "fetch_trigger"

    intent = state.get("intent") or {}
    intent_type = intent.get("intent_type", "SEARCH_PRODUCT") if isinstance(intent, dict) else "SEARCH_PRODUCT"

    if intent_type == "INVALID_QUERY":
        return "response"

    if intent_type in ("FEATURE_QUERY", "PRODUCT_DEEP_DIVE", "TECHNICAL_EXPLANATION"):
        return "product_knowledge"

    if intent_type == "FOLLOWUP_QUERY":
        # Route based on last turn's intent so follow-ups feel continuous
        session = state.get("session")
        last = getattr(session, "last_intent_type", "") if session else ""
        if isinstance(session, dict):
            last = session.get("last_intent_type", "")
        if last == "COMPARE_PRODUCTS":
            return "review_consensus"
        return "review_consensus"  # default follow-up → recommendation branch

    # All other intents go through review_consensus first
    return "review_consensus"


def _route_after_reviews(state: GraphState) -> str:
    """Branch after review_consensus to comparison or recommendation."""
    intent = state.get("intent") or {}
    intent_type = intent.get("intent_type", "SEARCH_PRODUCT") if isinstance(intent, dict) else "SEARCH_PRODUCT"

    if intent_type == "COMPARE_PRODUCTS":
        return "comparison_reasoning"
    return "recommendation"


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_graph(
    *,
    redis_client: Any,
    db_session: Any,
    llm_client: Any,
    embedding_provider: Any,
    celery_available: bool = True,
) -> Any:
    """Build and compile the LangGraph state machine.

    Call once per request (or cache the compiled graph per request context).
    Dependencies are injected via factory closures.
    """
    graph = StateGraph(GraphState)

    # Register all nodes
    graph.add_node("session_context", _make_session_context_node(redis_client))
    graph.add_node("intent_understanding", _make_intent_node(llm_client))
    graph.add_node("query_expansion", _make_query_expansion_node(llm_client, embedding_provider))
    graph.add_node("hybrid_retrieval", _make_hybrid_retrieval_node(db_session, redis_client))
    graph.add_node("review_consensus", _make_review_consensus_node(db_session, llm_client))
    graph.add_node("comparison_reasoning", _make_comparison_reasoning_node(llm_client))
    graph.add_node("recommendation", _make_recommendation_node(llm_client))
    graph.add_node("product_knowledge", _make_product_knowledge_node(db_session, llm_client))
    graph.add_node("fetch_trigger", _make_fetch_trigger_node(redis_client, celery_available))
    graph.add_node("structured_output", _make_structured_output_node(db_session))
    graph.add_node("session_update", _make_session_update_node(redis_client))
    graph.add_node("response", _make_response_node())

    # Entry point
    graph.set_entry_point("session_context")

    # Linear chain: session → intent → expand → retrieve
    graph.add_edge("session_context", "intent_understanding")
    graph.add_edge("intent_understanding", "query_expansion")
    graph.add_edge("query_expansion", "hybrid_retrieval")

    # Branch after retrieval
    graph.add_conditional_edges(
        "hybrid_retrieval",
        _route_after_retrieval,
        {
            "response": "response",              # cache_hit or INVALID_QUERY
            "fetch_trigger": "fetch_trigger",    # no DB data → fire Celery scrape
            "review_consensus": "review_consensus",  # compare or recommend
            "product_knowledge": "product_knowledge",  # feature query
        },
    )

    # Branch after review_consensus
    graph.add_conditional_edges(
        "review_consensus",
        _route_after_reviews,
        {
            "comparison_reasoning": "comparison_reasoning",
            "recommendation": "recommendation",
        },
    )

    # All reasoning branches converge to structured_output
    graph.add_edge("comparison_reasoning", "structured_output")
    graph.add_edge("recommendation", "structured_output")
    graph.add_edge("product_knowledge", "structured_output")

    # structured_output → validate → session_update → response
    graph.add_edge("structured_output", "session_update")
    graph.add_edge("session_update", "response")

    # Fetch trigger and cache hit → response directly
    graph.add_edge("fetch_trigger", "response")
    graph.add_edge("response", END)

    return graph.compile()


def compile_graph(
    *,
    redis_client: Any,
    db_session: Any,
    llm_client: Any,
    embedding_provider: Any,
    celery_available: bool = True,
) -> Any:
    """Alias for build_graph — used by API layer."""
    return build_graph(
        redis_client=redis_client,
        db_session=db_session,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        celery_available=celery_available,
    )
