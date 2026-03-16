"""LLM Model Router.

Maps agent types to appropriate model IDs.
Small/fast agents use gpt-4o-mini; reasoning-heavy agents use gpt-4o.
Prevents cost explosion from using large models everywhere.
"""

from __future__ import annotations

# Agent name → OpenAI model ID
MODEL_MAP: dict[str, str] = {
    "intent":               "gpt-4o-mini",
    "query_expansion":      "gpt-4o-mini",
    "product_knowledge":    "gpt-4o-mini",
    "review_consensus":     "gpt-4o",
    "recommendation":       "gpt-4o",
    "comparison_reasoning": "gpt-4o",
}

_DEFAULT_MODEL = "gpt-4o-mini"


def get_model_for_agent(agent_type: str) -> str:
    """Return the OpenAI model ID appropriate for the given agent type.

    Falls back to gpt-4o-mini for unknown agent names.
    """
    return MODEL_MAP.get(agent_type, _DEFAULT_MODEL)
