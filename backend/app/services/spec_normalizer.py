"""Spec Normalizer Service.

Scraped specs are inconsistent across sources. This service normalizes
raw specs_jsonb into a canonical dict that agents can reason on reliably.

Examples:
    "Battery: Li-Po 5000 mAh"  → battery_mah: 5000
    "Battery Capacity: 5,000 mAh" → battery_mah: 5000
    "Display: 6.7\" AMOLED 120Hz" → display_hz: 120, display_type: "AMOLED"

Called by: Recommendation Agent, Comparison Reasoning Agent, Query Expansion Agent.
"""

from __future__ import annotations

import re
from typing import Any


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_int(text: str, pattern: str) -> int | None:
    """Extract first integer matching the regex pattern from text."""
    m = re.search(pattern, str(text).replace(",", ""), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, str(text).replace(",", ""), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _search_keys(specs: dict, *keywords: str) -> str | None:
    """Return value of the first key containing any keyword (case-insensitive)."""
    for key, val in specs.items():
        if any(kw.lower() in key.lower() for kw in keywords):
            return str(val)
    return None


# ── Display type detection ────────────────────────────────────────────────────

_DISPLAY_TYPES = ["AMOLED", "Super AMOLED", "OLED", "IPS LCD", "LCD", "TFT", "LTPO"]


def _detect_display_type(text: str) -> str | None:
    for dt in _DISPLAY_TYPES:
        if dt.lower() in text.lower():
            return dt
    return None


# ── Chipset normalization ─────────────────────────────────────────────────────

_CHIPSET_PATTERNS = [
    r"(Snapdragon[\s\w]+)",
    r"(Apple [AM]\d+[\w\s]*)",
    r"(Dimensity\s+\d+[\w\s]*)",
    r"(Exynos\s+\d+[\w\s]*)",
    r"(Helio\s+[A-Z]\d+[\w\s]*)",
    r"(Kirin\s+\d+[\w\s]*)",
    r"(Intel Core\s+[\w\-]+)",
    r"(AMD Ryzen\s+[\w\s]+)",
]


def _extract_chipset(text: str) -> str | None:
    for pattern in _CHIPSET_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ── Main normalizer ────────────────────────────────────────────────────────────

def normalize_specs(specs_jsonb: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw specs_jsonb into canonical spec fields.

    Returns a dict with canonical keys alongside the original raw specs.
    Missing fields are None — never fabricated.
    """
    if not specs_jsonb:
        return {}

    canonical: dict[str, Any] = {}

    # Battery (mAh)
    battery_raw = _search_keys(specs_jsonb, "battery", "batt")
    if battery_raw:
        canonical["battery_mah"] = _extract_int(battery_raw, r"(\d+)\s*mah")

    # Display refresh rate (Hz)
    display_raw = _search_keys(specs_jsonb, "display", "screen", "refresh")
    if display_raw:
        canonical["display_hz"] = _extract_int(display_raw, r"(\d+)\s*hz")
        canonical["display_type"] = _detect_display_type(display_raw)

    # Chipset
    chipset_raw = _search_keys(specs_jsonb, "chipset", "processor", "cpu", "soc")
    if chipset_raw:
        canonical["chipset"] = _extract_chipset(chipset_raw) or chipset_raw.strip()

    # RAM (GB)
    ram_raw = _search_keys(specs_jsonb, "ram", "memory")
    if ram_raw:
        canonical["ram_gb"] = _extract_int(ram_raw, r"(\d+)\s*gb")

    # Storage (GB)
    storage_raw = _search_keys(specs_jsonb, "storage", "internal")
    if storage_raw:
        canonical["storage_gb"] = _extract_int(storage_raw, r"(\d+)\s*gb")

    # Main camera (MP)
    camera_raw = _search_keys(specs_jsonb, "main camera", "rear camera", "primary camera", "camera")
    if camera_raw:
        canonical["camera_mp_main"] = _extract_int(camera_raw, r"(\d+)\s*mp")

    # Front camera (MP)
    front_raw = _search_keys(specs_jsonb, "front camera", "selfie", "front")
    if front_raw:
        canonical["front_camera_mp"] = _extract_int(front_raw, r"(\d+)\s*mp")

    # Charging speed (W)
    charging_raw = _search_keys(specs_jsonb, "charging", "fast charge", "power delivery")
    if charging_raw:
        canonical["charging_w"] = _extract_int(charging_raw, r"(\d+)\s*w")

    # Weight (g)
    weight_raw = _search_keys(specs_jsonb, "weight")
    if weight_raw:
        canonical["weight_g"] = _extract_int(weight_raw, r"(\d+)\s*g")

    # Remove None values so downstream agents get clean data
    canonical = {k: v for k, v in canonical.items() if v is not None}

    return canonical


def extract_technical_fields(specs_jsonb: dict[str, Any]) -> dict[str, Any | None]:
    """Extract deep technical fields for ProductKnowledge agent.

    Returns a structured dict focusing on fields users ask about for
    deep product questions.
    """
    normalized = normalize_specs(specs_jsonb)

    # Battery chemistry
    battery_raw = _search_keys(specs_jsonb, "battery", "batt") or ""
    chemistry: str | None = None
    if "li-ion" in battery_raw.lower():
        chemistry = "Li-Ion"
    elif "li-po" in battery_raw.lower() or "lithium polymer" in battery_raw.lower():
        chemistry = "Li-Po"

    # Sensor size (camera)
    sensor_raw = _search_keys(specs_jsonb, "sensor size", "sensor") or ""
    sensor_size = sensor_raw.strip() if sensor_raw else None

    # Thermal design
    thermal_raw = _search_keys(specs_jsonb, "thermal", "cooling", "heat") or ""
    thermal_design = thermal_raw.strip() if thermal_raw else None

    return {
        "chipset": normalized.get("chipset"),
        "display_type": normalized.get("display_type"),
        "battery_mah": normalized.get("battery_mah"),
        "battery_chemistry": chemistry,
        "sensor_size": sensor_size,
        "thermal_design": thermal_design,
        "ram_gb": normalized.get("ram_gb"),
        "storage_gb": normalized.get("storage_gb"),
        "camera_mp_main": normalized.get("camera_mp_main"),
        "charging_w": normalized.get("charging_w"),
        "weight_g": normalized.get("weight_g"),
    }
