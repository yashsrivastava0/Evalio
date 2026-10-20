"""Unit tests — component score calculator correctness.

Verifies:
  - 7 scores always within [0.0, 1.0]
  - Battery unit guard (phone uses mAh, laptop uses Wh; neutral=0.5 when missing)
  - Price score is deterministic with budget_max=None (absolute mode)
  - Popularity score defers sentiment in Phase 5
  - CPU tier mapping resolves correctly
  - score_version matches SCORE_VERSION from scoring_config
  - CRITICAL: no import from app.services.ranking (cross-layer isolation test)
"""

from __future__ import annotations

import importlib
import sys
import pytest

from scraper.normalization.spec_normalizer import NormalizedSpecs
from scraper.scoring.component_score_calculator import compute_all_scores
from app.domain.scoring_config import SCORE_VERSION


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _smartphone_specs(**overrides) -> NormalizedSpecs:
    base = NormalizedSpecs(
        ram_gb=8,
        storage_gb=256,
        battery_mah=5000,
        battery_wh=None,
        display_inches=6.7,
        refresh_rate_hz=120,
        cpu_model="Snapdragon 8 Gen 2",
        camera_mp_main=50,
        build_material="Gorilla Glass Victus",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _laptop_specs(**overrides) -> NormalizedSpecs:
    base = NormalizedSpecs(
        ram_gb=16,
        storage_gb=512,
        battery_mah=None,
        battery_wh=72.0,
        display_inches=15.6,
        refresh_rate_hz=60,
        cpu_model="Core Ultra 7",
        camera_mp_main=None,
        build_material="aluminium frame",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ── Isolation: no cross-layer import ─────────────────────────────────────────


class TestNoRankingImport:
    """CRITICAL: component_score_calculator must NOT import from ranking.py."""

    def test_no_ranking_import(self):
        # Force re-import to get a clean module state
        mod_name = "scraper.scoring.component_score_calculator"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            mod = importlib.import_module(mod_name)

        # Walk all modules imported by component_score_calculator
        for name, value in vars(mod).items():
            if hasattr(value, "__module__"):
                assert "ranking" not in (value.__module__ or ""), (
                    f"Cross-layer violation: {name} is from 'ranking' module"
                )

        # Use AST to check actual import nodes in component_score_calculator.py.
        # This ignores comments and docstrings that name the forbidden module,
        # and avoids sys.modules which is session-wide and polluted by other tests.
        import ast
        import inspect
        try:
            source = inspect.getsource(mod)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "services.ranking" not in alias.name, (
                            f"Cross-layer violation: direct import of {alias.name!r}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "services.ranking" not in module, (
                        f"Cross-layer violation: from {module!r} import ..."
                    )
        except (OSError, TypeError):
            pass  # source not available (compiled) — skip


# ── Score range tests ─────────────────────────────────────────────────────────


class TestScoreRange:
    def test_all_scores_in_range_smartphone(self):
        specs = _smartphone_specs()
        result = compute_all_scores(specs, 49999.0, 1000, "smartphone")
        for field in [
            "performance_score", "battery_score", "camera_score",
            "display_score", "build_score", "price_score", "popularity_score"
        ]:
            val = getattr(result, field)
            assert 0.0 <= val <= 1.0, f"{field} out of range: {val}"

    def test_all_scores_in_range_laptop(self):
        specs = _laptop_specs()
        result = compute_all_scores(specs, 89999.0, 200, "laptop")
        for field in [
            "performance_score", "battery_score", "camera_score",
            "display_score", "build_score", "price_score", "popularity_score"
        ]:
            val = getattr(result, field)
            assert 0.0 <= val <= 1.0, f"{field} out of range: {val}"

    def test_all_scores_in_range_zero_data(self):
        """All-None specs must still return valid scores in range."""
        specs = NormalizedSpecs()
        result = compute_all_scores(specs, None, 0, "smartphone")
        for field in [
            "performance_score", "battery_score", "camera_score",
            "display_score", "build_score", "price_score", "popularity_score"
        ]:
            val = getattr(result, field)
            assert 0.0 <= val <= 1.0, f"{field} out of range on null input: {val}"


# ── Battery unit guard ────────────────────────────────────────────────────────


class TestBatteryUnitGuard:
    def test_phone_uses_mah(self):
        """Phone battery score uses mAh (not Wh)."""
        specs_small_mah = _smartphone_specs(battery_mah=2000, battery_wh=None)
        specs_large_mah = _smartphone_specs(battery_mah=6000, battery_wh=None)

        result_small = compute_all_scores(specs_small_mah, None, 0, "smartphone")
        result_large = compute_all_scores(specs_large_mah, None, 0, "smartphone")

        assert result_large.battery_score > result_small.battery_score

    def test_laptop_uses_wh(self):
        """Laptop battery score uses Wh (not mAh)."""
        specs_small_wh = _laptop_specs(battery_wh=40.0, battery_mah=None)
        specs_large_wh = _laptop_specs(battery_wh=100.0, battery_mah=None)

        result_small = compute_all_scores(specs_small_wh, None, 0, "laptop")
        result_large = compute_all_scores(specs_large_wh, None, 0, "laptop")

        assert result_large.battery_score > result_small.battery_score

    def test_neutral_when_both_missing(self):
        """Neutral score (0.5) when neither battery_mah nor battery_wh is available."""
        from app.domain.scoring_config import BATTERY_NEUTRAL
        specs = _smartphone_specs(battery_mah=None, battery_wh=None)
        result = compute_all_scores(specs, None, 0, "smartphone")
        assert result.battery_score == BATTERY_NEUTRAL

    def test_no_cross_unit_contamination(self):
        """Phone with only Wh and laptop with only mAh should still produce valid scores
        via the voltage approximation fallback — not blow up."""
        phone_specs = _smartphone_specs(battery_mah=None, battery_wh=15.0)  # ≈ 4054 mAh
        laptop_specs = _laptop_specs(battery_mah=6500, battery_wh=None)  # ≈ 72.15 Wh

        phone_result = compute_all_scores(phone_specs, None, 0, "smartphone")
        laptop_result = compute_all_scores(laptop_specs, None, 0, "laptop")

        assert 0.0 <= phone_result.battery_score <= 1.0
        assert 0.0 <= laptop_result.battery_score <= 1.0


# ── Price score — budget_max=None is deterministic ────────────────────────────


class TestPriceScoreDeterminism:
    def test_budget_none_is_deterministic(self):
        """Same price with budget_max=None always returns same score."""
        specs = _smartphone_specs()
        r1 = compute_all_scores(specs, 49999.0, 0, "smartphone", budget_max=None)
        r2 = compute_all_scores(specs, 49999.0, 0, "smartphone", budget_max=None)
        assert r1.price_score == r2.price_score

    def test_cheaper_product_higher_score(self):
        """Lower price → higher price_score when budget_max=None."""
        specs = _smartphone_specs()
        cheap = compute_all_scores(specs, 15000.0, 0, "smartphone", budget_max=None)
        expensive = compute_all_scores(specs, 100000.0, 0, "smartphone", budget_max=None)
        assert cheap.price_score > expensive.price_score

    def test_none_price_returns_zero(self):
        specs = _smartphone_specs()
        result = compute_all_scores(specs, None, 0, "smartphone")
        assert result.price_score == 0.0


# ── Popularity: Phase 5 sentiment deferrral ───────────────────────────────────


class TestPopularityPhase5:
    def test_no_reviews_returns_zero(self):
        specs = _smartphone_specs()
        result = compute_all_scores(specs, None, 0, "smartphone", sentiment_score=None)
        assert result.popularity_score == 0.0

    def test_more_reviews_higher_score(self):
        specs = _smartphone_specs()
        few = compute_all_scores(specs, None, 100, "smartphone", sentiment_score=None)
        many = compute_all_scores(specs, None, 5000, "smartphone", sentiment_score=None)
        assert many.popularity_score > few.popularity_score

    def test_sentiment_none_does_not_crash(self):
        """sentiment_score=None (Phase 5 default) must not raise."""
        specs = _smartphone_specs()
        result = compute_all_scores(specs, None, 1000, "smartphone", sentiment_score=None)
        assert 0.0 <= result.popularity_score <= 1.0


# ── Score version ─────────────────────────────────────────────────────────────


class TestScoreVersion:
    def test_score_version_matches_config(self):
        specs = _smartphone_specs()
        result = compute_all_scores(specs, None, 0, "smartphone")
        assert result.score_version == SCORE_VERSION

    def test_computed_at_is_set(self):
        specs = _smartphone_specs()
        result = compute_all_scores(specs, None, 0, "smartphone")
        assert result.computed_at is not None


# ── CPU tier mapping ──────────────────────────────────────────────────────────


class TestCpuTierMapping:
    def test_flagship_cpu_high_score(self):
        specs = _smartphone_specs(cpu_model="Snapdragon 8 Gen 3")
        result = compute_all_scores(specs, None, 0, "smartphone")
        assert result.performance_score >= 0.90

    def test_budget_cpu_low_score(self):
        specs = _smartphone_specs(cpu_model="Unisoc T616")
        result = compute_all_scores(specs, None, 0, "smartphone")
        assert result.performance_score <= 0.40

    def test_flagship_laptop_cpu_high_score(self):
        specs = _laptop_specs(cpu_model="Core Ultra 9")
        result = compute_all_scores(specs, None, 0, "laptop")
        assert result.performance_score >= 0.90

    def test_unknown_cpu_gets_default(self):
        specs = _smartphone_specs(cpu_model="VeryUnknownChip X1")
        result = compute_all_scores(specs, None, 0, "smartphone")
        # Default for unknown is 0.3 ± RAM bonus
        assert result.performance_score <= 0.5
