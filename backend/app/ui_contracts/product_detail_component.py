"""Product Detail UI Contract.

Maps ProductKnowledgeResult agent output → stable UI dict.
"""

from __future__ import annotations

from typing import Any


def to_ui(product_knowledge_result: dict[str, Any]) -> dict[str, Any]:
    """Map ProductKnowledgeResult → product_detail UI component."""
    data = product_knowledge_result.get("data", {})
    return {
        "component": "product_detail",
        "version": "v1",
        "id": data.get("product_id"),
        "slug": data.get("product_slug"),
        "title": data.get("canonical_title"),
        "technical": {
            "chipset": data.get("chipset"),
            "battery_mah": data.get("battery_mah"),
            "battery_chemistry": data.get("battery_chemistry"),
            "display_type": data.get("display_type"),
            "display_hz": data.get("display_hz"),
            "thermal_design": data.get("thermal_design"),
            "sensor_size": data.get("sensor_size"),
            "camera_mp_main": data.get("camera_mp_main"),
            "charging_w": data.get("charging_w"),
            "ram_gb": data.get("ram_gb"),
            "storage_gb": data.get("storage_gb"),
            "weight_g": data.get("weight_g"),
        },
        "feature_highlights": data.get("feature_highlights", []),
    }
