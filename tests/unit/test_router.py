"""Unit tests for the intent router.

Verifies:
    1. Rule-based classification for clear queries
    2. Category detection (phone vs laptop)
    3. Budget extraction (single value and range)
    4. Brand detection
    5. Priority detection (camera, battery, etc.)
    6. LLM fallback for ambiguous queries
"""

from __future__ import annotations

import pytest

from app.agents.router import classify_intent_rules, IntentPlan


# ── Category Detection ──

def test_phone_category():
    plan = classify_intent_rules("best phone under 30000")
    assert plan is not None
    assert plan.category == "smartphone"


def test_laptop_category():
    plan = classify_intent_rules("best laptop for coding")
    assert plan is not None
    assert plan.category == "laptop"


def test_iphone_maps_to_smartphone():
    plan = classify_intent_rules("compare iphone 15 vs samsung")
    assert plan is not None
    assert plan.category == "smartphone"


def test_macbook_maps_to_laptop():
    plan = classify_intent_rules("macbook air vs thinkpad")
    assert plan is not None
    assert plan.category == "laptop"


# ── Budget Extraction ──

def test_budget_under():
    plan = classify_intent_rules("phone under 25000")
    assert plan is not None
    assert plan.budget_max == 25000


def test_budget_range():
    plan = classify_intent_rules("phone 20000 to 30000")
    assert plan is not None
    assert plan.budget_min == 20000
    assert plan.budget_max == 30000


def test_budget_with_currency():
    plan = classify_intent_rules("phone under ₹35,000")
    assert plan is not None
    assert plan.budget_max == 35000


# ── Brand Detection ──

def test_single_brand():
    plan = classify_intent_rules("samsung phone under 20000")
    assert plan is not None
    assert "samsung" in plan.brands


def test_multiple_brands():
    plan = classify_intent_rules("oneplus vs samsung phone")
    assert plan is not None
    assert "oneplus" in plan.brands
    assert "samsung" in plan.brands


# ── Priority Detection ──

def test_camera_priority():
    plan = classify_intent_rules("best camera phone under 30000")
    assert plan is not None
    assert "camera" in plan.priority_list


def test_battery_priority():
    plan = classify_intent_rules("phone with best battery life")
    assert plan is not None
    assert "battery" in plan.priority_list


def test_gaming_maps_to_performance():
    plan = classify_intent_rules("best gaming phone")
    assert plan is not None
    assert "performance" in plan.priority_list


# ── Comparison Detection ──

def test_compare_intent():
    plan = classify_intent_rules("compare pixel 8 vs iphone 15")
    assert plan is not None
    assert plan.is_comparison is True


# ── Ambiguous → LLM fallback ──

def test_ambiguous_query_returns_none():
    """Truly ambiguous query should return None → LLM fallback."""
    plan = classify_intent_rules("what should I buy for my mom")
    assert plan is None


def test_classification_method():
    plan = classify_intent_rules("best samsung phone under 20000")
    assert plan is not None
    assert plan.classification_method == "rule"
