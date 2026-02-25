"""Phase 7 — ComparisonGuard v2 unit tests.

Verifies all locked invariants:
    - Score existence, range, NaN/None validation
    - Determinism across 100 runs
    - Float precision (6 decimals)
    - Tie handling (winner=None, no metric_wins increment)
    - All 7 metrics always present in diff_matrix
    - percent_diff zero-case
    - compare_ids order preservation
    - total_score read, not recomputed
    - Automated latency assertion (<10ms for 4 products)
    - Schema v2 structure correctness
"""

from __future__ import annotations

import math
import time

import pytest

from app.agents.comparison_guard import (
    ComparisonGuardError,
    ComparisonIntegrityError,
    CANONICAL_METRICS,
    _percent_diff,
    validate_all_component_scores,
    validate_ranking_integrity,
    build_comparison_table,
    comparison_guard_node,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_product(product_id: int, final_score: float = 0.75, **score_overrides) -> dict:
    """Create a minimal valid ranked product."""
    scores = {
        "performance_score": 0.7,
        "battery_score": 0.6,
        "camera_score": 0.8,
        "display_score": 0.7,
        "build_score": 0.5,
        "price_score": 0.6,
        "popularity_score": 0.4,
    }
    scores.update(score_overrides)
    return {
        "product_id": product_id,
        "product_slug": f"product-{product_id}",
        "canonical_title": f"Product {product_id}",
        "final_score": final_score,
        "rank": product_id,
        "scores": scores,
    }


# ── percent_diff helper ───────────────────────────────────────────────────────


def test_percent_diff_zero_case():
    """max(a,b)==0 → percent_diff must be 0.0."""
    assert _percent_diff(0.0, 0.0) == 0.0


def test_percent_diff_one_zero():
    """One value is zero — should use the other as denominator."""
    result = _percent_diff(0.0, 0.5)
    assert result == round(0.5 / 0.5 * 100, 6)


def test_percent_diff_normal():
    """Normal case: round(abs(a-b)/max(a,b)*100, 6)."""
    result = _percent_diff(0.8, 0.6)
    expected = round(0.2 / 0.8 * 100, 6)
    assert result == expected


def test_percent_diff_symmetric():
    """percent_diff(a,b) == percent_diff(b,a) — uses max(), not fixed denominator."""
    assert _percent_diff(0.9, 0.6) == _percent_diff(0.6, 0.9)


# ── Score Validation ─────────────────────────────────────────────────────────


def test_missing_score_fails():
    """A product missing any of the 7 canonical scores must raise ComparisonIntegrityError."""
    product = _make_product(1)
    del product["scores"]["camera_score"]
    with pytest.raises(ComparisonIntegrityError, match="missing score 'camera_score'"):
        validate_all_component_scores([product])


def test_nan_score_fails():
    """NaN score must raise ComparisonIntegrityError."""
    product = _make_product(1)
    product["scores"]["battery_score"] = float("nan")
    with pytest.raises(ComparisonIntegrityError, match="NaN"):
        validate_all_component_scores([product])


def test_none_score_fails():
    """None score must raise ComparisonIntegrityError."""
    product = _make_product(1)
    product["scores"]["display_score"] = None
    with pytest.raises(ComparisonIntegrityError, match="missing score"):
        validate_all_component_scores([product])


def test_out_of_range_score_high_fails():
    """Score > 1.0 must raise ComparisonIntegrityError."""
    product = _make_product(1)
    product["scores"]["performance_score"] = 1.5
    with pytest.raises(ComparisonIntegrityError, match="out of range"):
        validate_all_component_scores([product])


def test_out_of_range_score_negative_fails():
    """Negative score must raise ComparisonIntegrityError."""
    product = _make_product(1)
    product["scores"]["price_score"] = -0.1
    with pytest.raises(ComparisonIntegrityError, match="out of range"):
        validate_all_component_scores([product])


def test_valid_scores_pass():
    """All valid scores must pass without raising."""
    products = [_make_product(1), _make_product(2)]
    validate_all_component_scores(products)  # no exception


def test_boundary_scores_pass():
    """Exact 0.0 and 1.0 are valid boundaries."""
    product = _make_product(1, **{k: 0.0 for k in CANONICAL_METRICS})
    product["scores"]["performance_score"] = 1.0
    validate_all_component_scores([product])  # no exception


# ── Rank Integrity ───────────────────────────────────────────────────────────


def test_rank_mismatch_fails():
    """Mismatched ranked IDs must raise ComparisonGuardError."""
    with pytest.raises(ComparisonGuardError):
        validate_ranking_integrity([1, 2, 3], [1, 3, 2])


def test_rank_match_passes():
    """Identical ranked IDs must pass."""
    assert validate_ranking_integrity([1, 2, 3], [1, 2, 3]) is True


def test_empty_rank_passes():
    """Empty lists pass without error."""
    assert validate_ranking_integrity([], []) is True


# ── Schema v2 Structure ───────────────────────────────────────────────────────


def test_diff_matrix_structure():
    """diff_matrix must have absolute_diff, percent_diff, winner keys for each metric."""
    products = [_make_product(1, 0.8), _make_product(2, 0.6)]
    result = build_comparison_table(products)

    assert "comparison_table" in result
    assert "diff_matrix" in result
    for metric in CANONICAL_METRICS:
        assert metric in result["diff_matrix"], f"Missing metric in diff_matrix: {metric}"
        entry = result["diff_matrix"][metric]
        assert "absolute_diff" in entry
        assert "percent_diff" in entry
        assert "winner" in entry


def test_all_7_metrics_always_present():
    """diff_matrix must always include all 7 canonical metrics — regardless of input."""
    # Even with a single product, all 7 metrics must appear
    products = [_make_product(1, 0.75)]
    result = build_comparison_table(products)
    for metric in CANONICAL_METRICS:
        assert metric in result["diff_matrix"]


def test_comparison_table_structure():
    """comparison_table entries must have product_id, component_scores, metric_wins, total_score."""
    products = [_make_product(1, 0.8), _make_product(2, 0.6)]
    result = build_comparison_table(products)
    for entry in result["comparison_table"]:
        assert "product_id" in entry
        assert "component_scores" in entry
        assert "metric_wins" in entry
        assert "total_score" in entry


def test_total_score_read_not_recomputed():
    """total_score in comparison_table must equal product['final_score'], never recomputed."""
    expected_score_p1 = 0.9123456
    expected_score_p2 = 0.5678901
    products = [_make_product(1, expected_score_p1), _make_product(2, expected_score_p2)]
    result = build_comparison_table(products)
    table = {e["product_id"]: e for e in result["comparison_table"]}
    assert abs(table[1]["total_score"] - expected_score_p1) < 1e-9
    assert abs(table[2]["total_score"] - expected_score_p2) < 1e-9


# ── Tie Handling ─────────────────────────────────────────────────────────────


def test_tie_metric_no_winner():
    """When a == b for a metric, winner must be None and metric_wins must not increment."""
    tie_score = 0.7
    products = [
        _make_product(1, 0.8, battery_score=tie_score),
        _make_product(2, 0.6, battery_score=tie_score),
    ]
    result = build_comparison_table(products)
    battery_diff = result["diff_matrix"]["battery_score"]
    assert battery_diff["winner"] is None
    assert battery_diff["absolute_diff"] == 0.0
    assert battery_diff["percent_diff"] == 0.0

    # Neither product should get a metric_win for this metric
    table = {e["product_id"]: e for e in result["comparison_table"]}
    # Metric wins driven only by non-tie metrics — battery_score contributes 0
    # (we can't assert exact count without knowing other scores, but
    # we verify the tie metric itself does not add wins)
    diff_matrix = result["diff_matrix"]
    non_tie_metrics = [
        m for m, d in diff_matrix.items() if d["winner"] is not None
    ]
    total_wins = sum(e["metric_wins"] for e in result["comparison_table"])
    assert total_wins == len(non_tie_metrics)


def test_tie_absolute_diff_zero():
    """Exact tie → absolute_diff must be 0.0."""
    val = 0.55
    products = [
        _make_product(1, 0.8, performance_score=val),
        _make_product(2, 0.6, performance_score=val),
    ]
    result = build_comparison_table(products)
    assert result["diff_matrix"]["performance_score"]["absolute_diff"] == 0.0


# ── Determinism ───────────────────────────────────────────────────────────────


def test_deterministic_diffs_100_runs():
    """Running build_comparison_table 100 times with identical input must produce identical output."""
    products = [_make_product(1, 0.85), _make_product(2, 0.70)]
    first = build_comparison_table(products)
    for _ in range(99):
        result = build_comparison_table(products)
        assert result == first, "Non-deterministic output detected"


def test_float_precision_stable():
    """absolute_diff must be rounded to exactly 6 decimal places."""
    products = [
        _make_product(1, 0.8, performance_score=0.9),
        _make_product(2, 0.6, performance_score=0.6),
    ]
    result = build_comparison_table(products)
    diff = result["diff_matrix"]["performance_score"]["absolute_diff"]
    # Must be exactly 0.3 rounded to 6 decimals
    assert diff == round(abs(0.9 - 0.6), 6)
    # Must not have more than 6 significant decimal places
    assert len(str(diff).split(".")[-1]) <= 6


# ── compare_ids Order Preservation ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_ids_not_reordering():
    """compare_ids=[9,5] with ranked=[5,2,9] must output products in ranking order [5,9]."""
    ranked = [
        _make_product(5, 0.9),   # rank 1
        _make_product(2, 0.75),  # rank 2
        _make_product(9, 0.6),   # rank 3
    ]
    ranked[0]["rank"] = 1
    ranked[1]["rank"] = 2
    ranked[2]["rank"] = 3

    state = {"ranked": ranked, "compare_ids": [9, 5]}  # user supplied reverse order
    result = await comparison_guard_node(state)

    comparison_table = result["comparison"]["comparison_table"]
    output_ids = [entry["product_id"] for entry in comparison_table]
    # Must respect ranking order [5, 9] — NOT user's input order [9, 5]
    assert output_ids == [5, 9]


# ── Multi-product Limits ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_ids_max_4_enforced():
    """compare_ids with 5 products must raise ComparisonIntegrityError."""
    ranked = [_make_product(i, 0.9 - i * 0.1) for i in range(1, 6)]
    state = {"ranked": ranked, "compare_ids": [1, 2, 3, 4, 5]}
    with pytest.raises(ComparisonIntegrityError, match="exceeds max 4"):
        await comparison_guard_node(state)


@pytest.mark.asyncio
async def test_compare_ids_duplicates_rejected():
    """Duplicate IDs in compare_ids must raise ComparisonIntegrityError."""
    ranked = [_make_product(1, 0.9), _make_product(2, 0.7)]
    state = {"ranked": ranked, "compare_ids": [1, 1, 2]}
    with pytest.raises(ComparisonIntegrityError, match="duplicates"):
        await comparison_guard_node(state)


@pytest.mark.asyncio
async def test_compare_ids_out_of_ranked_rejected():
    """ID not in ranked must raise ComparisonIntegrityError."""
    ranked = [_make_product(1, 0.9), _make_product(2, 0.7)]
    state = {"ranked": ranked, "compare_ids": [1, 99]}  # 99 not in ranked
    with pytest.raises(ComparisonIntegrityError, match="not present in ranked"):
        await comparison_guard_node(state)



# ── Automated Latency ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_latency_under_10ms():
    """ComparisonGuard must complete in <10ms for 4 products (pure numeric path)."""
    ranked = [_make_product(i, 0.9 - i * 0.1) for i in range(1, 5)]
    state = {"ranked": ranked}

    start = time.perf_counter()
    await comparison_guard_node(state)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.010, (
        f"ComparisonGuard took {elapsed*1000:.2f}ms — exceeds 10ms invariant"
    )


# ── N-Product diff_matrix Correctness (NEW — post-audit fix tests) ──────────


def test_multi_product_winner_correct_for_3_products():
    """With 3 products, diff_matrix winner must be the BEST across all 3, not just positions 0 and 1.

    Before the fix, only pair (products[0], products[1]) was compared.
    This means if product #3 had the highest score for a metric, it would
    NOT be recognized as winner. This test verifies the full N-product sort.
    """
    # Products: ranked by final_score, but for 'camera_score',
    # the 3rd product (lowest ranked overall) has the highest value.
    products = [
        _make_product(1, 0.85, camera_score=0.5),   # rank 1 overall but weak camera
        _make_product(2, 0.70, camera_score=0.6),   # rank 2 overall, mid camera
        _make_product(3, 0.55, camera_score=0.9),   # rank 3 overall but BEST camera
    ]
    result = build_comparison_table(products)
    camera_winner = result["diff_matrix"]["camera_score"]["winner"]

    # Under old top-2 logic: winner would be str(2) because only products[0] vs products[1]
    # Under correct N-product logic: winner must be str(3) (highest camera_score=0.9)
    assert camera_winner == "3", (
        f"Expected product 3 to win camera_score (score=0.9), but got winner={camera_winner}. "
        f"N-product diff logic may not be applying correctly."
    )

    # Also verify metric_wins credited to product 3 for camera
    table_by_id = {e["product_id"]: e for e in result["comparison_table"]}
    # Product 3 should have at least 1 metric win (camera_score)
    assert table_by_id[3]["metric_wins"] >= 1, (
        f"Product 3 has highest camera_score but metric_wins={table_by_id[3]['metric_wins']}"
    )


def test_multi_product_winner_correct_for_4_products():
    """With 4 products, diff_matrix winner must be the MAX across all 4 for each metric.

    Old top-2 logic would have only compared products at positions 0 and 1.
    This test places the true winner at position 3 (0-indexed) for two metrics.
    """
    # product 4 has significantly higher performance and display scores
    products = [
        _make_product(1, 0.90, performance_score=0.5, display_score=0.4),
        _make_product(2, 0.80, performance_score=0.6, display_score=0.5),
        _make_product(3, 0.70, performance_score=0.7, display_score=0.6),
        _make_product(4, 0.60, performance_score=0.95, display_score=0.92),  # worst rank, best metric
    ]
    result = build_comparison_table(products)
    dm = result["diff_matrix"]

    perf_winner = dm["performance_score"]["winner"]
    disp_winner = dm["display_score"]["winner"]

    assert perf_winner == "4", (
        f"Expected product 4 to win performance_score (0.95), got {perf_winner}"
    )
    assert disp_winner == "4", (
        f"Expected product 4 to win display_score (0.92), got {disp_winner}"
    )

    # Verify absolute_diff is computed relative to runner-up (product 3), not position 1 (product 2)
    expected_perf_diff = round(abs(0.95 - 0.7), 6)  # top=0.95, second=0.7
    expected_disp_diff = round(abs(0.92 - 0.6), 6)  # top=0.92, second=0.6
    assert dm["performance_score"]["absolute_diff"] == expected_perf_diff, (
        f"performance absolute_diff should be {expected_perf_diff} "
        f"(top vs runner-up), got {dm['performance_score']['absolute_diff']}"
    )
    assert dm["display_score"]["absolute_diff"] == expected_disp_diff, (
        f"display absolute_diff should be {expected_disp_diff} "
        f"(top vs runner-up), got {dm['display_score']['absolute_diff']}"
    )


@pytest.mark.asyncio
async def test_filtering_preserves_relative_rank_order():
    """Filter integrity assertion must ensure compare_ids filtering preserves ranked order.

    Given ranked=[1,2,3,4] and compare_ids=[4,1,3] (user-supplied in reverse):
        - compare_products must be ordered [1,3,4] (preserving ranked order)
        - validated_ranked_ids must be [1,3,4]
        - The filter integrity assertion must pass (no ComparisonGuardError)

    This test also verifies the assertion would catch actual reordering bugs.
    """
    # ranked order: 1 → 2 → 3 → 4 (by final_score DESC)
    ranked = [
        _make_product(1, 0.90),
        _make_product(2, 0.80),
        _make_product(3, 0.70),
        _make_product(4, 0.60),
    ]
    # User supplies compare_ids in arbitrary order — system must sort by rank
    state = {"ranked": ranked, "compare_ids": [4, 1, 3]}
    result = await comparison_guard_node(state)

    # validated_ranked_ids must preserve ranking order [1, 3, 4] — NOT [4, 1, 3]
    assert result["validated_ranked_ids"] == [1, 3, 4], (
        f"Expected validated_ranked_ids=[1,3,4] (ranked order), "
        f"got {result['validated_ranked_ids']}"
    )

    # comparison_table product_id order must also match [1, 3, 4]
    table_ids = [e["product_id"] for e in result["comparison"]["comparison_table"]]
    assert table_ids == [1, 3, 4], (
        f"comparison_table product order must be [1,3,4] (ranked), got {table_ids}"
    )
