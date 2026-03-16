"""Session Context Agent — first LangGraph node.

Loads the user's session from Redis and attaches it to GraphState.
Must run BEFORE the Intent Router.

Redis key: session:<session_id>
Stored fields:
    - active_comparison: product slugs currently being compared
    - last_results: top candidates from last query
    - session_products: all product slugs seen this session
    - conversation_history: last 5 turns (FIFO)
    - user_preferences: inferred {focus, budget_max}
    - last_intent_type, last_query: for follow-up resolution
    - turn_count: number of turns in this session
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SessionData:
    """Structured session data loaded from Redis."""

    session_id: str = ""
    active_comparison: list[str] = field(default_factory=list)
    last_results: list[dict[str, Any]] = field(default_factory=list)

    # AI Decision Engine fields
    session_products: list[str] = field(default_factory=list)
    """All product slugs seen across the session (capped at 20, for follow-up resolution)."""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    """Last 5 turns: [{query, intent_type, top_products}] — FIFO capped."""
    user_preferences: dict[str, Any] = field(default_factory=dict)
    """Inferred preferences: {focus: [...], budget_max: N}."""
    last_intent_type: str = ""
    last_query: str = ""

    # Legacy fields (preserved for Redis round-trip compatibility)
    conversation_summary: str = ""
    last_filter_hash: str = ""
    last_query_embedding_hash: str = ""
    turn_count: int = 0


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


async def load_session(redis_client: Any, session_id: str) -> SessionData:
    """Load session from Redis. Returns empty session if not found."""
    key = _session_key(session_id)
    raw = await redis_client.get(key)

    if raw is None:
        logger.info("session_not_found", session_id=session_id)
        return SessionData(session_id=session_id)

    try:
        data = orjson.loads(raw)
        return SessionData(
            session_id=session_id,
            active_comparison=data.get("active_comparison", []),
            last_results=data.get("last_results", []),
            session_products=data.get("session_products", []),
            conversation_history=data.get("conversation_history", []),
            user_preferences=data.get("user_preferences", {}),
            last_intent_type=data.get("last_intent_type", ""),
            last_query=data.get("last_query", ""),
            conversation_summary=data.get("conversation_summary", ""),
            last_filter_hash=data.get("last_filter_hash", ""),
            last_query_embedding_hash=data.get("last_query_embedding_hash", ""),
            turn_count=data.get("turn_count", 0),
        )
    except Exception as e:
        logger.error("session_load_error", session_id=session_id, error=str(e))
        return SessionData(session_id=session_id)


async def save_session(redis_client: Any, session: SessionData, ttl_seconds: int = 3600) -> None:
    """Write session data back to Redis."""
    key = _session_key(session.session_id)
    # Cap conversation_history at 5 entries
    history: list[dict[str, Any]] = list(session.conversation_history)
    if len(history) > 5:
        start = len(history) - 5
        history = [history[i] for i in range(start, len(history))]
    payload = orjson.dumps({
        "active_comparison": session.active_comparison,
        "last_results": session.last_results,
        "session_products": session.session_products,
        "conversation_history": history,
        "user_preferences": session.user_preferences,
        "last_intent_type": session.last_intent_type,
        "last_query": session.last_query,
        "conversation_summary": session.conversation_summary,
        "last_filter_hash": session.last_filter_hash,
        "last_query_embedding_hash": session.last_query_embedding_hash,
        "turn_count": session.turn_count,
    })
    await redis_client.set(key, payload, ex=ttl_seconds)
    logger.info("session_saved", session_id=session.session_id, turn=session.turn_count)


# ── LangGraph Node Function ──

async def session_context_node(state: dict, *, redis_client: Any) -> dict:
    """LangGraph node: loads session and returns state update."""
    session_id = state.get("session_id", "")
    session = await load_session(redis_client, session_id)
    return {"session": session}
