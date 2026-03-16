"""Intent Understanding Agent — Tool functions."""

from __future__ import annotations

from typing import Any


def get_session_context(session: Any) -> dict[str, Any]:
    """Extract relevant session context for follow-up resolution."""
    if session is None:
        return {}
    return {
        "session_products": getattr(session, "session_products", []),
        "last_query": getattr(session, "last_query", ""),
        "last_intent_type": getattr(session, "last_intent_type", ""),
        "turn_count": getattr(session, "turn_count", 0),
        "active_comparison": getattr(session, "active_comparison", []),
    }


def format_conversation_history(session: Any) -> str:
    """Format last 3 conversation turns for follow-up resolution.

    Returns a compact text representation for the LLM prompt.
    """
    if session is None:
        return ""

    history = getattr(session, "conversation_history", []) or []
    if not history:
        return ""

    # Take last 3 turns
    recent = history[-3:]
    lines: list[str] = []
    for i, turn in enumerate(recent, 1):
        query = turn.get("query", "")
        intent = turn.get("intent_type", "")
        products = turn.get("top_products", [])
        products_str = ", ".join(products[:3]) if products else "none"
        lines.append(f"Turn {i}: query='{query}' intent={intent} products=[{products_str}]")

    return "\n".join(lines)


def build_intent_prompt_context(query: str, session: Any) -> dict[str, str]:
    """Build all prompt template variables for the intent agent."""
    ctx = get_session_context(session)
    history_text = format_conversation_history(session)

    return {
        "query": query,
        "session_history": history_text or "No previous conversation.",
        "session_products": ", ".join(ctx.get("session_products", [])) or "none",
        "last_query": ctx.get("last_query", "") or "none",
        "last_intent": ctx.get("last_intent_type", "") or "none",
        "turn_count": str(ctx.get("turn_count", 0)),
    }
