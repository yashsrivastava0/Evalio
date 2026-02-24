"""Unit tests — slug generation and variant dict building.

Tests:
  - Slug format: brand-model-category-year
  - Cross-category uniqueness
  - Cross-year uniqueness
  - Collision safety for identical models in different categories
  - Price parser edge cases (₹ format, commas, unavailable, ranges)
"""

from __future__ import annotations

import pytest
from scraper.normalization.variant_builder import generate_slug, _extract_release_year
from scraper.playwright_scraper.base_scraper import BaseScraper


# ── Release year extraction ───────────────────────────────────────────────────

class TestReleaseYearExtraction:
    def test_iso_date(self):
        assert _extract_release_year("2023-09-12") == 2023

    def test_month_year(self):
        assert _extract_release_year("September 2023") == 2023

    def test_year_only(self):
        assert _extract_release_year("2023") == 2023

    def test_none_input(self):
        assert _extract_release_year(None) is None

    def test_no_year_found(self):
        assert _extract_release_year("Unknown Date") is None

    def test_old_year_not_matched(self):
        # Only matches 20xx years
        assert _extract_release_year("15 March 1999") is None


# ── Slug generation ───────────────────────────────────────────────────────────

class TestSlugGeneration:
    def test_basic_slug(self):
        slug = generate_slug("Apple", "iPhone 15 Pro", "smartphone", 2023)
        assert slug == "apple-iphone-15-pro-smartphone-2023"

    def test_na_suffix_when_no_year(self):
        slug = generate_slug("Apple", "iPhone 15 Pro", "smartphone", None)
        assert slug.endswith("-na")

    def test_cross_category_uniqueness(self):
        """Same brand+model with different category → different slug."""
        slug_phone = generate_slug("Samsung", "Galaxy S24", "smartphone", 2024)
        slug_laptop = generate_slug("Samsung", "Galaxy S24", "laptop", 2024)
        assert slug_phone != slug_laptop

    def test_cross_year_uniqueness(self):
        """Same brand+model with different year → different slug."""
        slug_23 = generate_slug("Apple", "iPhone 15", "smartphone", 2023)
        slug_24 = generate_slug("Apple", "iPhone 15", "smartphone", 2024)
        assert slug_23 != slug_24

    def test_lowercase_output(self):
        slug = generate_slug("SAMSUNG", "GALAXY S24 ULTRA", "SMARTPHONE", 2024)
        assert slug == slug.lower()

    def test_no_special_characters(self):
        slug = generate_slug("OnePlus", "12R (Glacier Green)", "smartphone", 2024)
        import re
        assert re.match(r"^[a-z0-9-]+$", slug), f"Slug contains invalid chars: {slug}"

    def test_slug_max_length(self):
        """Slug must not exceed 128 characters."""
        long_model = "A" * 200
        slug = generate_slug("Brand", long_model, "smartphone", 2024)
        assert len(slug) <= 128

    def test_unicode_normalized(self):
        """Accented chars are stripped to ASCII."""
        slug = generate_slug("Huawei", "Mate 60 Pró", "smartphone", 2023)
        assert "ó" not in slug
        assert "pro" in slug


# ── Price parser ──────────────────────────────────────────────────────────────

class TestPriceParser:
    """Test BaseScraper._parse_price_inr static method."""

    def test_rupee_symbol_with_commas(self):
        """₹1,24,999 → 124999.0"""
        assert BaseScraper._parse_price_inr("₹1,24,999") == 124999.0

    def test_plain_number(self):
        assert BaseScraper._parse_price_inr("49999") == 49999.0

    def test_rs_prefix(self):
        assert BaseScraper._parse_price_inr("Rs. 49,999") == 49999.0

    def test_unavailable_returns_none(self):
        assert BaseScraper._parse_price_inr("Currently unavailable") is None

    def test_out_of_stock_returns_none(self):
        assert BaseScraper._parse_price_inr("Out of Stock") is None

    def test_price_range_returns_none(self):
        """Ambiguous ranges must not produce a value."""
        assert BaseScraper._parse_price_inr("₹10,000 - ₹15,000") is None

    def test_empty_string_returns_none(self):
        assert BaseScraper._parse_price_inr("") is None

    def test_none_input_returns_none(self):
        assert BaseScraper._parse_price_inr(None) is None

    def test_negative_price_returns_none(self):
        assert BaseScraper._parse_price_inr("-500") is None

    def test_absurd_price_returns_none(self):
        """Price > ₹1 crore = likely a parsing error."""
        assert BaseScraper._parse_price_inr("₹99,99,99,999") is None

    def test_with_decimal(self):
        assert BaseScraper._parse_price_inr("₹1,099.99") == 1099.99

    def test_plain_commas_only(self):
        assert BaseScraper._parse_price_inr("1,00,000") == 100000.0
