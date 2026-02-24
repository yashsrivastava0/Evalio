"""Unit tests — spec normalization correctness.

Tests that spec_normalizer correctly extracts typed values from
raw scraper key-value strings, including edge cases.
"""

from __future__ import annotations

import pytest
from scraper.normalization.spec_normalizer import normalize_specs, NormalizedSpecs


class TestRamParsing:
    def test_gb_form(self):
        specs = normalize_specs({"RAM": "8 GB LPDDR5"})
        assert specs.ram_gb == 8

    def test_gb_no_space(self):
        specs = normalize_specs({"RAM": "12GB"})
        assert specs.ram_gb == 12

    def test_mb_form_converted(self):
        specs = normalize_specs({"Memory": "8192 MB"})
        assert specs.ram_gb == 8

    def test_mb_below_gb_threshold_returns_none(self):
        # 512 MB < 1024 → not a valid GB value
        specs = normalize_specs({"Memory": "512 MB"})
        assert specs.ram_gb is None

    def test_unknown_key_ignored(self):
        specs = normalize_specs({"Quantum Flux": "8 GB"})
        assert specs.ram_gb is None


class TestStorageParsing:
    def test_gb_storage(self):
        specs = normalize_specs({"Internal Storage": "256 GB UFS 3.1"})
        assert specs.storage_gb == 256

    def test_tb_converted(self):
        specs = normalize_specs({"SSD": "1 TB"})
        assert specs.storage_gb == 1024

    def test_fractional_tb(self):
        specs = normalize_specs({"Storage": "2 TB"})
        assert specs.storage_gb == 2048

    def test_tiny_gb_rejected(self):
        # "4 GB" is too small for storage — should not parse as storage
        specs = normalize_specs({"Hard Disk": "4 GB"})
        assert specs.storage_gb is None


class TestBatteryParsing:
    def test_mah_phone(self):
        specs = normalize_specs({"Battery": "5000 mAh"})
        assert specs.battery_mah == 5000
        assert specs.battery_wh is None

    def test_wh_laptop(self):
        specs = normalize_specs({"Battery Capacity": "72 Wh"})
        assert specs.battery_wh == 72.0
        assert specs.battery_mah is None

    def test_fractional_wh(self):
        specs = normalize_specs({"Battery": "72.6 Wh"})
        assert specs.battery_wh == 72.6

    def test_no_unit_returns_none(self):
        specs = normalize_specs({"Battery": "5000"})  # no unit
        assert specs.battery_mah is None
        assert specs.battery_wh is None


class TestDisplayParsing:
    def test_inch_symbol(self):
        specs = normalize_specs({"Display Size": '6.7"'})
        assert specs.display_inches == 6.7

    def test_inch_text(self):
        specs = normalize_specs({"Screen Size": "15.6 inches"})
        assert specs.display_inches == 15.6

    def test_out_of_range_rejected(self):
        specs = normalize_specs({"Display": "100 inches"})
        assert specs.display_inches is None


class TestRefreshRateParsing:
    def test_120hz(self):
        specs = normalize_specs({"Refresh Rate": "120Hz AMOLED"})
        assert specs.refresh_rate_hz == 120

    def test_165hz(self):
        specs = normalize_specs({"Screen Refresh Rate": "165 hz"})
        assert specs.refresh_rate_hz == 165


class TestCpuExtraction:
    def test_keeps_longer_string(self):
        specs = normalize_specs({
            "Processor": "Snapdragon 8",
            "CPU Model": "Snapdragon 8 Gen 3 (3.3 GHz)",
        })
        # Should prefer the longer description
        assert "Gen 3" in (specs.cpu_model or "")


class TestWeightParsing:
    def test_grams(self):
        specs = normalize_specs({"Item Weight": "190g"})
        assert specs.weight_g == 190.0

    def test_kg(self):
        specs = normalize_specs({"Weight": "1.5 kg"})
        assert specs.weight_g == 1500.0


class TestCameraParsing:
    def test_mp(self):
        specs = normalize_specs({"Rear Camera": "108 MP OIS"})
        assert specs.camera_mp_main == 108

    def test_megapixels_text(self):
        specs = normalize_specs({"Primary Camera": "50 megapixels"})
        assert specs.camera_mp_main == 50

    def test_above_range_rejected(self):
        specs = normalize_specs({"Camera Resolution": "600 MP"})
        assert specs.camera_mp_main is None


class TestMissingFields:
    def test_completely_empty(self):
        specs = normalize_specs({})
        assert specs == NormalizedSpecs()

    def test_all_empty_strings(self):
        specs = normalize_specs({"RAM": "", "Battery": ""})
        assert specs.ram_gb is None
        assert specs.battery_mah is None
