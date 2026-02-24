"""Deterministic spec normalizer — raw scraper dict → NormalizedSpecs.

No LLM. No hallucination. Missing field → None.
All parsing uses regex against known string patterns.

Extracts both battery_mah (phones) and battery_wh (laptops) separately
so component_score_calculator can use the correct unit per category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Normalized spec dataclass ─────────────────────────────────────────────────


@dataclass
class NormalizedSpecs:
    """Fully parsed, unit-normalized product specs.

    All fields nullable — prefer None over a wrong value.
    """

    # Memory & storage
    ram_gb: int | None = None
    storage_gb: int | None = None

    # Power
    battery_mah: int | None = None          # phones
    battery_wh: float | None = None         # laptops

    # Display
    display_inches: float | None = None
    refresh_rate_hz: int | None = None

    # Processor
    cpu_model: str | None = None
    gpu_model: str | None = None

    # Camera
    camera_mp_main: int | None = None

    # Physical
    weight_g: float | None = None
    build_material: str | None = None       # raw string, scored in calculator

    # Launch
    launch_date_str: str | None = None


# ── Regex helpers ─────────────────────────────────────────────────────────────


def _first_match(pattern: str, text: str, group: int = 1) -> str | None:
    """Return first regex group match or None."""
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(group).strip() if m else None


def _parse_ram(raw: str) -> int | None:
    """Parse RAM: '8 GB', '8GB', '8192 MB', '12 GB LPDDR5' → int GB."""
    raw = raw.strip()

    # MB → convert to GB
    m = re.search(r"(\d+)\s*mb", raw, re.IGNORECASE)
    if m:
        mb = int(m.group(1))
        return mb // 1024 if mb >= 1024 else None

    # GB form
    m = re.search(r"(\d+)\s*gb", raw, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def _parse_storage(raw: str) -> int | None:
    """Parse storage: '256 GB SSD', '512GB UFS', '1 TB' → int GB."""
    # TB form
    m = re.search(r"(\d+(?:\.\d+)?)\s*tb", raw, re.IGNORECASE)
    if m:
        return int(float(m.group(1)) * 1024)

    # GB form
    m = re.search(r"(\d+)\s*gb", raw, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        # Sanity: storage should be ≥ 8 GB and power-of-two-ish
        if val >= 8:
            return val

    return None


def _parse_battery_mah(raw: str) -> int | None:
    """Parse mAh battery: '4500 mAh', '4500mAh'. Returns None if not mAh."""
    m = re.search(r"(\d+)\s*mah", raw, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_battery_wh(raw: str) -> float | None:
    """Parse Wh battery: '72 Wh', '72Wh', '72.6 Wh'. Returns None if not Wh."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*wh", raw, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_display_inches(raw: str) -> float | None:
    """Parse screen size: '6.7\"', '6.7 inch', '15.6 inches', '15.6-inch'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[\"'\u2033]?\s*(?:inch(?:es)?|-inch)?", raw, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        # Sanity: display between 3" and 20"
        if 3.0 <= val <= 20.0:
            return val
    return None


def _parse_refresh_rate(raw: str) -> int | None:
    """Parse refresh rate: '120Hz', '120 Hz', '120 HZ AMOLED'."""
    m = re.search(r"(\d+)\s*hz", raw, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 30 <= val <= 500:
            return val
    return None


def _parse_weight(raw: str) -> float | None:
    """Parse weight: '190g', '190 grams', '1.5 kg'."""
    # kg form
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg", raw, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000

    # g form
    m = re.search(r"(\d+(?:\.\d+)?)\s*g(?:rams?|$|\s)", raw, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 50 <= val <= 5000:
            return val

    return None


def _parse_camera_mp(raw: str) -> int | None:
    """Parse camera megapixels: '108MP', '108 MP', '50 megapixels'."""
    m = re.search(r"(\d+)\s*(?:mp|megapixels?)", raw, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 500:
            return val
    return None


# ── Key alias maps ────────────────────────────────────────────────────────────
# Maps common scraper key variants → canonical parser function

_KEY_ALIASES: dict[str, str] = {
    # RAM
    "ram": "ram", "memory": "ram", "ram capacity": "ram",
    "system memory": "ram", "installed memory": "ram",
    # Storage
    "storage": "storage", "internal storage": "storage",
    "rom": "storage", "hard disk size": "storage", "ssd": "storage",
    "hard disk": "storage", "flash storage": "storage",
    # Battery
    "battery": "battery", "battery capacity": "battery",
    "battery size": "battery", "charge capacity": "battery",
    # Display
    "display size": "display", "screen size": "display",
    "display": "display", "screen": "display",
    # Refresh rate
    "refresh rate": "refresh", "screen refresh rate": "refresh",
    "display refresh rate": "refresh",
    # CPU
    "processor": "cpu", "cpu": "cpu", "chipset": "cpu",
    "processor brand": "cpu", "cpu model": "cpu",
    # GPU
    "gpu": "gpu", "graphics": "gpu", "graphics processor": "gpu",
    # Weight
    "weight": "weight", "item weight": "weight",
    "product weight": "weight",
    # Camera
    "rear camera": "camera", "primary camera": "camera",
    "back camera": "camera", "main camera": "camera",
    "camera resolution": "camera",
    # Build
    "back material": "build", "body material": "build",
    "build material": "build", "material": "build",
    "construction": "build", "frame material": "build",
    # Launch date
    "date first available": "launch", "launch date": "launch",
    "date available": "launch", "release date": "launch",
}


def _canonical_key(raw_key: str) -> str | None:
    """Map a scraper key string to its canonical category, or None if unknown."""
    key = raw_key.lower().strip().rstrip(":")
    return _KEY_ALIASES.get(key)


# ── Main normalizer ───────────────────────────────────────────────────────────


def normalize_specs(raw_specs: dict[str, str]) -> NormalizedSpecs:
    """Convert raw scraper key-value dict to NormalizedSpecs.

    Deterministic. No LLM. Any missing or un-parseable field → None.
    Never mutates the input dict.

    Args:
        raw_specs: Raw {label: value} dict from scraper (string values).

    Returns:
        NormalizedSpecs dataclass with parsed, typed fields.
    """
    specs = NormalizedSpecs()

    for raw_key, raw_val in raw_specs.items():
        if not raw_val or not str(raw_val).strip():
            continue

        val = str(raw_val).strip()
        canon = _canonical_key(raw_key)

        if canon == "ram":
            specs.ram_gb = specs.ram_gb or _parse_ram(val)

        elif canon == "storage":
            specs.storage_gb = specs.storage_gb or _parse_storage(val)

        elif canon == "battery":
            # Try both units; store whichever parses
            if specs.battery_mah is None:
                specs.battery_mah = _parse_battery_mah(val)
            if specs.battery_wh is None:
                specs.battery_wh = _parse_battery_wh(val)

        elif canon == "display":
            specs.display_inches = specs.display_inches or _parse_display_inches(val)

        elif canon == "refresh":
            specs.refresh_rate_hz = specs.refresh_rate_hz or _parse_refresh_rate(val)

        elif canon == "cpu":
            # Keep longer (more descriptive) cpu_model if already found
            if specs.cpu_model is None or len(val) > len(specs.cpu_model):
                specs.cpu_model = val

        elif canon == "gpu":
            if specs.gpu_model is None or len(val) > len(specs.gpu_model):
                specs.gpu_model = val

        elif canon == "weight":
            specs.weight_g = specs.weight_g or _parse_weight(val)

        elif canon == "camera":
            specs.camera_mp_main = specs.camera_mp_main or _parse_camera_mp(val)

        elif canon == "build":
            if specs.build_material is None:
                specs.build_material = val

        elif canon == "launch":
            if specs.launch_date_str is None:
                specs.launch_date_str = val

    return specs
