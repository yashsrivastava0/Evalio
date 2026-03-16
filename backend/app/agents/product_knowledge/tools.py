"""Product Knowledge Agent — Tool functions."""

from __future__ import annotations

from typing import Any

from app.tools.db_tools import fetch_product_specs as _fetch_specs
from app.tools.spec_tools import get_technical_fields


async def fetch_product_specs(db: Any, product_id: int) -> dict[str, Any]:
    """Fetch full product specs from DB."""
    return await _fetch_specs(db, product_id)


def extract_technical_fields(specs_jsonb: dict[str, Any]) -> dict[str, Any]:
    """Extract deep technical fields from raw specs."""
    return get_technical_fields(specs_jsonb)
