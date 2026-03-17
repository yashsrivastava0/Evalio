"""Product Knowledge Agent — LangGraph node.

Fetches deep technical specs for a product and uses LLM to extract
and explain the most relevant features for the user's query.

Replaces the old _make_feature_lookup_node in graph.py.
Uses gpt-4o-mini — spec extraction from structured data is straightforward.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.product_knowledge.schema import ProductKnowledgeResult, ProductDetail
from app.agents.product_knowledge.tools import fetch_product_specs, extract_technical_fields
from app.core.logging import get_logger
from app.services.llm_router import get_model_for_agent

logger = get_logger(__name__)

_PROMPT_PATH = pathlib.Path(__file__).parent / "prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_MAX_RETRIES = 2


async def product_knowledge_node(
    state: dict[str, Any],
    *,
    db_session: AsyncSession,
    llm_client: Any,
) -> dict[str, Any]:
    """LangGraph node: fetch and explain deep technical product details.

    Returns state update:
        product_knowledge_result: ProductKnowledgeResult (as dict)
    """
    query = state.get("query", "")
    candidates = state.get("candidates", [])
    model = get_model_for_agent("product_knowledge")

    if not candidates:
        logger.warning("product_knowledge_no_candidates")
        return {"product_knowledge_result": {}}

    # Use top candidate for feature queries
    top = candidates[0]
    product_id = top.get("product_id", 0)
    product_slug = top.get("product_slug", "")
    canonical_title = top.get("canonical_title", "Unknown")

    # Fetch full specs from DB (or use what's already in state)
    specs_in_state = top.get("specs") or {}
    if specs_in_state:
        specs_data = {
            "product_id": product_id,
            "product_slug": product_slug,
            "canonical_title": canonical_title,
            "specs_jsonb": specs_in_state,
        }
    else:
        specs_data = await fetch_product_specs(db_session, product_id)

    specs_jsonb = specs_data.get("specs_jsonb") or {}
    technical = extract_technical_fields(specs_jsonb)

    # Format specs for LLM
    specs_text = "\n".join(f"  {k}: {v}" for k, v in (specs_jsonb or {}).items() if v)[:2000]

    user_message = (
        f"User question: \"{query}\"\n\n"
        f"Product: {canonical_title} (slug: {product_slug}, id: {product_id})\n\n"
        f"Pre-extracted technical fields:\n"
        f"  chipset: {technical.get('chipset')}\n"
        f"  battery_mah: {technical.get('battery_mah')}\n"
        f"  battery_chemistry: {technical.get('battery_chemistry')}\n"
        f"  display_type: {technical.get('display_type')}\n"
        f"  display_hz: {technical.get('display_hz')}\n"
        f"  thermal_design: {technical.get('thermal_design')}\n"
        f"  sensor_size: {technical.get('sensor_size')}\n"
        f"  charging_w: {technical.get('charging_w')}\n"
        f"  weight_g: {technical.get('weight_g')}\n\n"
        f"Full raw specs:\n{specs_text}\n\n"
        f"Extract the most relevant technical details for the user's question. "
        f"Return ONLY valid JSON matching the output schema."
    )

    result: ProductKnowledgeResult | None = None
    last_error = ""

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1200,
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            result = ProductKnowledgeResult(**parsed)
            logger.info("product_knowledge_complete", product_slug=product_slug)
            break

        except Exception as e:
            last_error = str(e)
            logger.warning("product_knowledge_error", attempt=attempt, error=last_error)
            if attempt == _MAX_RETRIES:
                logger.error("product_knowledge_fallback", error=last_error)
                # Fallback: use pre-extracted technical fields
                result = ProductKnowledgeResult(
                    data=ProductDetail(
                        product_id=product_id,
                        product_slug=product_slug,
                        canonical_title=canonical_title,
                        chipset=technical.get("chipset"),
                        sensor_size=technical.get("sensor_size"),
                        battery_chemistry=technical.get("battery_chemistry"),
                        display_type=technical.get("display_type"),
                        thermal_design=technical.get("thermal_design"),
                        battery_mah=technical.get("battery_mah"),
                        display_hz=technical.get("display_hz"),
                        ram_gb=technical.get("ram_gb"),
                        storage_gb=technical.get("storage_gb"),
                        camera_mp_main=technical.get("camera_mp_main"),
                        charging_w=technical.get("charging_w"),
                        weight_g=technical.get("weight_g"),
                        full_specs=specs_jsonb,
                        feature_highlights=[],
                    )
                )

    return {"product_knowledge_result": result.model_dump() if result else {}}
