"""Component score calculator — 7-factor deterministic scoring for scraped products.

All scores: 0.0 – 1.0 range. Deterministic. No LLM.

Imports:
  - app.domain.scoring_utils  → compute_price_score, compute_popularity_score
  - app.domain.scoring_config → versioned tier maps + SCORE_VERSION
  NEVER imports from app.services.ranking

Phase 5 sentiment rule:
  compute_popularity_score is called with sentiment_score=None.
  This produces a neutral-baseline score until the Phase 6 review_summary
  pipeline provides real sentiment values.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone

# ── Guard: reject any LLM imports at module level ─────────────────────────────
_FORBIDDEN_MODULES = {"openai", "langchain", "anthropic", "langchain_core", "langchain_community"}
for _m in list(sys.modules.keys()):
    if any(_m == _f or _m.startswith(_f + ".") for _f in _FORBIDDEN_MODULES):
        raise ImportError(
            f"LLM module '{_m}' must not be imported in component_score_calculator. "
            "Scoring is deterministic — no LLM allowed."
        )

# ── Domain imports (never ranking.py) ────────────────────────────────────────
# These are the only external imports allowed beyond stdlib and scoring_config.
import sys as _sys  # already imported above

# Lazy path adjustment for running from scraper/ directory context
import os
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_backend_path = os.path.join(_repo_root, "backend")
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)

from app.domain.scoring_config import (  # noqa: E402
    SCORE_VERSION,
    SMARTPHONE_CPU_TIERS,
    LAPTOP_CPU_TIERS,
    SMARTPHONE_RAM_BONUS,
    LAPTOP_RAM_BONUS,
    BUILD_MATERIAL_SCORES,
    BUILD_MATERIAL_DEFAULT,
    REFRESH_RATE_SCORES,
    REFRESH_RATE_DEFAULT,
    BATTERY_PHONE_MAX_MAH,
    BATTERY_LAPTOP_MAX_WH,
    BATTERY_NEUTRAL,
    CAMERA_MAX_MP,
)
from app.domain.scoring_utils import compute_price_score, compute_popularity_score  # noqa: E402


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ComponentScoreResult:
    """7-score result from a single compute_all_scores call.

    All scores are floats in [0.0, 1.0].
    score_version is tied to SCORE_VERSION constant in scoring_config.
    """

    performance_score: float
    battery_score: float
    camera_score: float
    display_score: float
    build_score: float
    price_score: float
    popularity_score: float
    score_version: int
    computed_at: datetime


# ── Individual score functions ────────────────────────────────────────────────


def _score_performance(
    cpu_model: str | None,
    ram_gb: int | None,
    category: str,
) -> float:
    """Score performance from CPU tier map + RAM bonus."""
    base_score = 0.3  # default for unknown CPUs

    tiers = SMARTPHONE_CPU_TIERS if category == "smartphone" else LAPTOP_CPU_TIERS
    cpu_lower = (cpu_model or "").lower()

    for keywords, score in tiers:
        if any(kw in cpu_lower for kw in keywords):
            base_score = score
            break

    # RAM bonus
    ram_bonus = 0.0
    if ram_gb is not None:
        if category == "smartphone":
            for threshold, bonus in sorted(SMARTPHONE_RAM_BONUS.items(), reverse=True):
                t = int(threshold.rstrip("+"))
                if (threshold.endswith("+") and ram_gb >= t) or (not threshold.endswith("+") and ram_gb == t):
                    ram_bonus = bonus
                    break
        else:
            for threshold, bonus in sorted(LAPTOP_RAM_BONUS.items(), reverse=True):
                t = int(threshold.rstrip("+"))
                if (threshold.endswith("+") and ram_gb >= t) or (not threshold.endswith("+") and ram_gb == t):
                    ram_bonus = bonus
                    break

    return round(min(1.0, max(0.0, base_score + ram_bonus)), 8)


def _score_battery(
    battery_mah: int | None,
    battery_wh: float | None,
    category: str,
) -> float:
    """Score battery using correct unit per category.

    Phone: mAh / BATTERY_PHONE_MAX_MAH (6000)
    Laptop: Wh / BATTERY_LAPTOP_MAX_WH (100)
    Missing both: BATTERY_NEUTRAL (0.5) — prevents wrong-unit scoring.
    """
    if category == "smartphone":
        if battery_mah is not None and battery_mah > 0:
            return round(min(1.0, battery_mah / BATTERY_PHONE_MAX_MAH), 8)
        # Fallback: Wh approximation (1 mAh ≈ 0.0037 Wh at 3.7V)
        if battery_wh is not None and battery_wh > 0:
            approx_mah = battery_wh / 0.0037
            return round(min(1.0, approx_mah / BATTERY_PHONE_MAX_MAH), 8)
    elif category == "laptop":
        if battery_wh is not None and battery_wh > 0:
            return round(min(1.0, battery_wh / BATTERY_LAPTOP_MAX_WH), 8)
        # Fallback: mAh at 11.1V typical laptop voltage
        if battery_mah is not None and battery_mah > 0:
            approx_wh = battery_mah * 11.1 / 1000
            return round(min(1.0, approx_wh / BATTERY_LAPTOP_MAX_WH), 8)

    return BATTERY_NEUTRAL


def _score_camera(camera_mp_main: int | None) -> float:
    """Score camera from main sensor megapixels."""
    if camera_mp_main is None or camera_mp_main <= 0:
        return 0.3  # neutral baseline; camera not applicable for laptops
    return round(min(1.0, camera_mp_main / CAMERA_MAX_MP), 8)


def _score_display(
    refresh_rate_hz: int | None,
    display_inches: float | None,
    category: str,
) -> float:
    """Score display from refresh rate + size factor."""
    # Refresh rate score
    refresh_score = REFRESH_RATE_DEFAULT
    if refresh_rate_hz is not None:
        for min_hz, score in REFRESH_RATE_SCORES:
            if refresh_rate_hz >= min_hz:
                refresh_score = score
                break

    # Size bonus: larger display gets a small boost (max 0.1)
    size_bonus = 0.0
    if display_inches is not None:
        if category == "smartphone":
            # Sweet spot: 6.1"–6.7"
            if 6.1 <= display_inches <= 6.7:
                size_bonus = 0.05
            elif display_inches > 6.7:
                size_bonus = 0.02  # too large is not always better
        elif category == "laptop":
            if display_inches >= 15.6:
                size_bonus = 0.05
            elif display_inches >= 13.3:
                size_bonus = 0.02

    return round(min(1.0, max(0.0, refresh_score + size_bonus)), 8)


def _score_build(build_material: str | None) -> float:
    """Score build quality from material tier map."""
    if not build_material:
        return BUILD_MATERIAL_DEFAULT
    lower = build_material.lower()
    for keywords, score in BUILD_MATERIAL_SCORES:
        if any(kw in lower for kw in keywords):
            return score
    return BUILD_MATERIAL_DEFAULT


# ── Main entry point ──────────────────────────────────────────────────────────


def compute_all_scores(
    specs: object,             # NormalizedSpecs
    price_inr: float | None,
    review_count: int,
    category: str,
    *,
    budget_max: float | None = None,   # None at scrape-time (no user query context)
    sentiment_score: float | None = None,  # None in Phase 5
) -> ComponentScoreResult:
    """Compute all 7 component scores for a product.

    Args:
        specs: NormalizedSpecs instance from spec_normalizer.
        price_inr: Best variant price in INR.
        review_count: Number of reviews scraped.
        category: "smartphone" or "laptop".
        budget_max: User query budget. None = absolute pricing mode (scrape-time).
        sentiment_score: Review sentiment [0,1]. None → Phase 5 neutral default.

    Returns:
        ComponentScoreResult with all 7 scores + version + timestamp.

    Note on budget_max=None:
        At scrape-time there is no user query context. compute_price_score
        handles budget_max=None deterministically via absolute reference ceiling
        (₹2,00,000). This is intentional and safe. See scoring_utils.py.
    """
    performance = _score_performance(
        getattr(specs, "cpu_model", None),
        getattr(specs, "ram_gb", None),
        category,
    )
    battery = _score_battery(
        getattr(specs, "battery_mah", None),
        getattr(specs, "battery_wh", None),
        category,
    )
    camera = _score_camera(getattr(specs, "camera_mp_main", None))
    display = _score_display(
        getattr(specs, "refresh_rate_hz", None),
        getattr(specs, "display_inches", None),
        category,
    )
    build = _score_build(getattr(specs, "build_material", None))
    price = compute_price_score(price_inr, budget_max)
    popularity = compute_popularity_score(review_count, sentiment_score)

    return ComponentScoreResult(
        performance_score=performance,
        battery_score=battery,
        camera_score=camera,
        display_score=display,
        build_score=build,
        price_score=price,
        popularity_score=popularity,
        score_version=SCORE_VERSION,
        computed_at=datetime.now(timezone.utc),
    )
