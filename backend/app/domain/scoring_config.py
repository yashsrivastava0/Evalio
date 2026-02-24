"""Versioned scoring configuration — all tier maps live here.

SCORE_VERSION must be bumped whenever any tier map changes.
The component_scores table stores score_version alongside computed_at.
A version mismatch triggers a recompute cycle.

Architectural rule:
  - This file is the ONLY place where scoring thresholds are defined.
  - ranking.py, component_score_calculator.py both read from here.
  - No hardcoded thresholds anywhere else.
"""

from __future__ import annotations

# ── Version ──────────────────────────────────────────────────────────────────
# Bump this integer whenever ANY tier map below changes.
# Triggers recompute for all products with score_version < SCORE_VERSION.

SCORE_VERSION: int = 1

# ── Performance: CPU Tier Maps ────────────────────────────────────────────────
# Smartphones — keyed by lowercase substring match against cpu_model field.
# Ordered highest → lowest; first match wins.

SMARTPHONE_CPU_TIERS: list[tuple[list[str], float]] = [
    # Tier S — flagship (score ≥ 0.90)
    (["snapdragon 8 gen 3", "snapdragon 8+ gen 2", "apple a17", "apple a18",
      "dimensity 9300", "dimensity 9200", "exynos 2400"], 0.95),
    # Tier A — upper mid (score ≥ 0.75)
    (["snapdragon 8 gen 2", "snapdragon 8 gen 1", "apple a16", "apple a15",
      "dimensity 9000", "dimensity 8300", "exynos 2200"], 0.85),
    # Tier B — mid-range
    (["snapdragon 7s gen 3", "snapdragon 7 gen 3", "snapdragon 7 gen 2",
      "dimensity 8200", "dimensity 8100", "helio g99", "exynos 1380"], 0.65),
    # Tier C — entry
    (["snapdragon 6 gen", "snapdragon 4 gen", "dimensity 7200",
      "dimensity 7050", "dimensity 700", "helio g85", "helio g88"], 0.40),
    # Tier D — budget
    (["unisoc", "helio a", "snapdragon 4", "snapdragon 680"], 0.20),
]

# Laptops — keyed by lowercase substring match against cpu_model field.

LAPTOP_CPU_TIERS: list[tuple[list[str], float]] = [
    # Tier S — flagship
    (["core ultra 9", "ryzen 9 7945hx", "m3 max", "m3 ultra",
      "m2 max", "m2 ultra", "m3 pro"], 0.95),
    # Tier A — performance
    (["core ultra 7", "core i9", "core i7-13", "core i7-14",
      "ryzen 9 7940", "ryzen 9 6900", "m3", "m2 pro", "m2"], 0.80),
    # Tier B — mainstream
    (["core ultra 5", "core i7-12", "core i5-13", "core i5-14",
      "ryzen 7 7740", "ryzen 7 7735", "ryzen 5 7640", "m1 pro"], 0.65),
    # Tier C — entry
    (["core i5-12", "core i5-11", "ryzen 5 7530", "ryzen 5 5600",
      "m1"], 0.45),
    # Tier D — budget
    (["celeron", "pentium", "ryzen 3", "core i3"], 0.25),
]

# RAM bonus: added on top of CPU tier score (capped at 1.0 total)
SMARTPHONE_RAM_BONUS: dict[str, float] = {
    "16+": 0.05,
    "12": 0.03,
    "8": 0.00,
    "6": -0.05,
    "4": -0.10,
}

LAPTOP_RAM_BONUS: dict[str, float] = {
    "64+": 0.05,
    "32": 0.03,
    "16": 0.00,
    "8":  -0.05,
}

# ── Build Material Tiers ──────────────────────────────────────────────────────
# Keyed by lowercase substring match; first match wins.

BUILD_MATERIAL_SCORES: list[tuple[list[str], float]] = [
    # Premium: glass back + metal frame
    (["gorilla glass", "ceramic glass", "titanium", "aluminium frame",
      "aluminum frame", "stainless steel"], 1.0),
    # Glass + plastic frame
    (["glass back", "glass"], 0.75),
    # Matte polycarbonate
    (["polycarbonate", "matte finish", "rubberized"], 0.55),
    # Standard plastic
    (["plastic", "ag glass"], 0.35),
]

# Default score when build_material is unknown
BUILD_MATERIAL_DEFAULT: float = 0.5

# ── Display Refresh Rate → Score ──────────────────────────────────────────────

REFRESH_RATE_SCORES: list[tuple[int, float]] = [
    # (minimum_hz, score)  — sorted descending, first match wins
    (165, 1.00),
    (144, 0.95),
    (120, 0.85),
    (90,  0.70),
    (75,  0.55),
    (60,  0.40),
]
REFRESH_RATE_DEFAULT: float = 0.30  # below 60 or unknown

# ── Battery Thresholds ────────────────────────────────────────────────────────

BATTERY_PHONE_MAX_MAH: float = 6000.0   # 6000 mAh → 1.0
BATTERY_LAPTOP_MAX_WH: float = 100.0    # 100 Wh → 1.0
BATTERY_NEUTRAL: float = 0.5            # used when unit unavailable

# ── Camera ────────────────────────────────────────────────────────────────────

CAMERA_MAX_MP: float = 200.0            # 200 MP → 1.0
