"""Shared scoring utility functions — imported by ranking.py and scraper scoring.

ARCHITECTURAL RULE:
  This module is the ONLY place compute_price_score and compute_popularity_score
  are defined. ranking.py and component_score_calculator.py both import from here.
  NEVER import from app.services.ranking inside this module.

Phase 5 note on popularity_score:
  sentiment_score is intentionally ignored (defaults to 0.5) until the
  review_summary pipeline is implemented in Phase 6. This keeps the score
  deterministic across scrape-time and rank-time without review data.
"""

from __future__ import annotations

import math


def compute_price_score(
    price_inr: float | None,
    budget_max: float | None = None,
) -> float:
    """Compute normalized price score (0.0 – 1.0).

    Lower price relative to budget = higher score.

    When budget_max is None (scrape-time, no user query context):
      Uses a reference ceiling of ₹2,00,000 to produce an absolute
      value score. This is deterministic and query-independent.

    When budget_max is supplied (ranking-time):
      Score = how much budget headroom remains, with a slight
      boost for cheaper options.

    Args:
        price_inr: Product price in INR. None or ≤0 → 0.0.
        budget_max: Optional upper budget from user intent. None = absolute mode.

    Returns:
        float in [0.0, 1.0]
    """
    if price_inr is None or price_inr <= 0:
        return 0.0

    if budget_max and budget_max > 0:
        # Relative to user budget — slight boost for cheaper options
        score = max(0.0, 1.0 - (price_inr / budget_max))
        return min(1.0, score * 1.5)

    # Absolute mode: inverse-normalize against ₹2,00,000 ceiling
    ref_max = 200_000.0
    return max(0.0, 1.0 - (price_inr / ref_max))


def compute_popularity_score(
    review_count: int = 0,
    sentiment_score: float | None = None,
) -> float:
    """Compute popularity score (0.0 – 1.0) from review data.

    Formula: log-scaled review count × sentiment multiplier.

    Phase 5 behaviour:
      sentiment_score is accepted as a parameter but the effective
      multiplier is capped at 0.5 when sentiment is None (neutral baseline).
      Once the Phase 6 review_summary pipeline writes real sentiment values,
      passing the actual score will yield the full range automatically.

    Args:
        review_count: Number of reviews (0+ integer).
        sentiment_score: Optional float in [0, 1]. None → 0.5 neutral.

    Returns:
        float in [0.0, 1.0]
    """
    count_score = (
        min(1.0, math.log1p(review_count) / 10.0) if review_count > 0 else 0.0
    )
    # Phase 5: sentiment defaults to 0.5 (neutral) when not yet available
    sentiment = sentiment_score if sentiment_score is not None else 0.5
    return round(count_score * sentiment, 8)
