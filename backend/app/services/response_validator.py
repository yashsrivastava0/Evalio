"""Response Validator — Guardrail Layer.

Verifies agent outputs against DB ground truth before sending to the UI.
Prevents hallucinated product slugs, prices, or specs from reaching users.

Strategy:
    1. Verify all product_slugs in the output exist in the DB.
    2. Strip or flag components that reference non-existent products.
    3. Non-blocking: on DB error, pass the output through unchanged.

Called by: graph.py structured_output node (after all agents complete).
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.tools.db_tools import fetch_products_by_slugs

logger = get_logger(__name__)


def _extract_slugs_from_output(output: dict[str, Any]) -> list[str]:
    """Extract all product_slug values from the structured output components."""
    slugs: set[str] = set()

    for component in output.get("components", []):
        data = component.get("data", {})
        component_type = component.get("component", "")

        if component_type == "recommendation_cards":
            for item in data.get("ranked", []):
                slug = item.get("product_slug")
                if slug:
                    slugs.add(slug)

        elif component_type == "comparison_table":
            for product in data.get("products", []):
                slug = product.get("product_slug")
                if slug:
                    slugs.add(slug)

        elif component_type == "product_detail":
            slug = data.get("product_slug")
            if slug:
                slugs.add(slug)

        elif component_type == "review_summary":
            for slug in data.get("summaries", {}).keys():
                slugs.add(slug)

    return list(slugs)


async def verify_agent_output(
    output: dict[str, Any],
    db: Any,
) -> dict[str, Any]:
    """Verify agent output against DB. Returns validated (possibly stripped) output.

    On any error, passes the output through unchanged (fail-open).
    """
    if not output or not output.get("components"):
        return output

    try:
        slugs_in_output = _extract_slugs_from_output(output)

        if not slugs_in_output:
            return output

        # Check which slugs actually exist in DB
        existing = await fetch_products_by_slugs(db, slugs_in_output)
        existing_slugs = {r["product_slug"] for r in existing}

        # Find hallucinated slugs
        hallucinated = set(slugs_in_output) - existing_slugs
        if not hallucinated:
            return output  # All good

        logger.warning(
            "response_validator_hallucination_detected",
            hallucinated_slugs=list(hallucinated),
        )

        # Strip hallucinated entries from components
        cleaned_components: list[dict[str, Any]] = []
        for component in output.get("components", []):
            cleaned = _strip_hallucinated(component, hallucinated)
            if cleaned:
                cleaned_components.append(cleaned)

        return {**output, "components": cleaned_components}

    except Exception as e:
        # Fail-open: return output unchanged on validator error
        logger.warning("response_validator_error", error=str(e))
        return output


def _strip_hallucinated(
    component: dict[str, Any],
    hallucinated_slugs: set[str],
) -> dict[str, Any] | None:
    """Remove hallucinated products from a component. Returns None if component is now empty."""
    component_type = component.get("component", "")
    data = component.get("data", {})

    if component_type == "recommendation_cards":
        ranked = [
            r for r in data.get("ranked", [])
            if r.get("product_slug") not in hallucinated_slugs
        ]
        if not ranked:
            return None
        return {**component, "data": {**data, "ranked": ranked}}

    elif component_type == "comparison_table":
        products = [
            p for p in data.get("products", [])
            if p.get("product_slug") not in hallucinated_slugs
        ]
        if len(products) < 2:
            return None  # Can't show a comparison with < 2 products
        return {**component, "data": {**data, "products": products}}

    elif component_type == "product_detail":
        if data.get("product_slug") in hallucinated_slugs:
            return None
        return component

    elif component_type == "review_summary":
        summaries = {
            slug: summary
            for slug, summary in data.get("summaries", {}).items()
            if slug not in hallucinated_slugs
        }
        if not summaries:
            return None
        return {**component, "data": {**data, "summaries": summaries}}

    return component
