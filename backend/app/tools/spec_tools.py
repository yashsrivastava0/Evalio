"""Spec tool functions shared across agents."""

from __future__ import annotations

from typing import Any

from app.services.spec_normalizer import normalize_specs, extract_technical_fields


def get_normalized_specs(specs_jsonb: dict[str, Any]) -> dict[str, Any]:
    """Return canonical normalized specs for reasoning agents."""
    return normalize_specs(specs_jsonb)


def get_technical_fields(specs_jsonb: dict[str, Any]) -> dict[str, Any]:
    """Return deep technical fields for ProductKnowledge agent."""
    return extract_technical_fields(specs_jsonb)


def format_specs_for_prompt(specs_jsonb: dict[str, Any], canonical: dict[str, Any]) -> str:
    """Format specs into a concise LLM-readable string.

    Uses canonical values where available, falls back to raw specs.
    """
    lines: list[str] = []

    if canonical.get("chipset"):
        lines.append(f"Chipset: {canonical['chipset']}")
    if canonical.get("ram_gb"):
        lines.append(f"RAM: {canonical['ram_gb']}GB")
    if canonical.get("storage_gb"):
        lines.append(f"Storage: {canonical['storage_gb']}GB")
    if canonical.get("display_type") or canonical.get("display_hz"):
        display_parts = []
        if canonical.get("display_type"):
            display_parts.append(canonical["display_type"])
        if canonical.get("display_hz"):
            display_parts.append(f"{canonical['display_hz']}Hz")
        lines.append(f"Display: {' '.join(display_parts)}")
    if canonical.get("camera_mp_main"):
        lines.append(f"Main camera: {canonical['camera_mp_main']}MP")
    if canonical.get("front_camera_mp"):
        lines.append(f"Front camera: {canonical['front_camera_mp']}MP")
    if canonical.get("battery_mah"):
        lines.append(f"Battery: {canonical['battery_mah']}mAh")
    if canonical.get("charging_w"):
        lines.append(f"Charging: {canonical['charging_w']}W")
    if canonical.get("weight_g"):
        lines.append(f"Weight: {canonical['weight_g']}g")

    return "\n".join(lines) if lines else "Specs not available"
