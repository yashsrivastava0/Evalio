"""Product Knowledge Agent — I/O schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductDetail(BaseModel):
    """Deep technical product data extracted by the LLM from raw specs."""

    product_id: int
    product_slug: str
    canonical_title: str

    # Technical fields (None if not available in specs)
    chipset: str | None = None
    sensor_size: str | None = None
    battery_chemistry: str | None = None
    display_type: str | None = None
    thermal_design: str | None = None

    # Normalized numeric fields
    battery_mah: int | None = None
    display_hz: int | None = None
    ram_gb: int | None = None
    storage_gb: int | None = None
    camera_mp_main: int | None = None
    charging_w: int | None = None
    weight_g: int | None = None

    full_specs: dict = Field(default_factory=dict, description="Raw specs_jsonb")
    feature_highlights: list[str] = Field(
        default_factory=list,
        description="2-4 most relevant features for the user's query",
    )


class ProductKnowledgeResult(BaseModel):
    """Output schema for the Product Knowledge Agent."""

    component: Literal["product_detail"] = "product_detail"
    ui_hint: Literal["render as list"] = "render as list"
    data: ProductDetail
