"""Candidate Cap Optimization Experiment.

Tests performance impact of different vector candidate limits.
Does NOT modify production config defaults.

Usage:
    python -m app.experiments.candidate_cap_experiment
"""

from __future__ import annotations

import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.ranking import rank_candidates

logger = get_logger(__name__)


@dataclass
class CapExperimentResult:
    """Result from a single cap experiment run."""
    sql_cap: int
    vector_cap: int
    ranking_cap: int
    dataset_size: int
    sql_filter_ms: float
    vector_rerank_ms: float
    ranking_ms: float
    total_ms: float
    sql_candidates: int
    vector_candidates: int
    final_ranked: int
    memory_peak_kb: float
    ranking_deterministic: bool


def generate_mock_products(count: int) -> list[dict[str, Any]]:
    """Generate mock products for cap testing."""
    products = []
    for i in range(1, count + 1):
        products.append({
            "product_id": i,
            "category": "smartphone",
            "brand": random.choice(["samsung", "apple", "oneplus", "xiaomi"]),
            "price_inr": random.randint(10_000, 100_000),
            "ram_gb": random.choice([4, 6, 8, 12]),
            "storage_gb": random.choice([64, 128, 256]),
            "availability": True,
            "scores": {
                "performance_score": round(random.uniform(0.1, 1.0), 4),
                "battery_score": round(random.uniform(0.1, 1.0), 4),
                "camera_score": round(random.uniform(0.1, 1.0), 4),
                "display_score": round(random.uniform(0.1, 1.0), 4),
                "build_score": round(random.uniform(0.1, 1.0), 4),
                "price_score": round(random.uniform(0.1, 1.0), 4),
                "popularity_score": round(random.uniform(0.0, 1.0), 4),
            },
            "embedding": [round(random.gauss(0, 1), 4) for _ in range(128)],
        })
    return products


def simulate_sql_filter(products: list[dict], max_candidates: int) -> list[int]:
    """Simulate SQL filter with cap."""
    filtered = [p for p in products if p["availability"] and p["price_inr"] <= 60_000]
    return [p["product_id"] for p in filtered[:max_candidates]]


def simulate_vector_rerank(
    products: list[dict],
    filtered_ids: list[int],
    query_emb: list[float],
    max_candidates: int,
) -> list[dict]:
    """Simulate vector re-rank on filtered subset."""
    import math

    id_set = set(filtered_ids)
    subset = [p for p in products if p["product_id"] in id_set]

    def cos_dist(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return 1.0 - (dot / (na * nb)) if na > 0 and nb > 0 else 1.0

    ranked = [{"product_id": p["product_id"], "distance": cos_dist(query_emb, p["embedding"])} for p in subset]
    ranked.sort(key=lambda x: x["distance"])
    return ranked[:max_candidates]


def build_ranking_candidates(products: list[dict], vector_ranked: list[dict], max_candidates: int) -> list[dict]:
    """Build ranking input from vector results."""
    pmap = {p["product_id"]: p for p in products}
    result = []
    for c in vector_ranked[:max_candidates]:
        pid = c["product_id"]
        if pid in pmap:
            p = pmap[pid]
            result.append({
                "product_id": pid,
                "product_slug": f"product-{pid}",
                "brand": p["brand"],
                "model_name": f"Model {pid}",
                "canonical_title": f"Product {pid}",
                "category": p["category"],
                "vector_distance": c["distance"],
                "specs": {},
                "variants": [{"price_inr": p["price_inr"], "availability": True, "vendor": "test"}],
                "scores": p["scores"],
            })
    return result


def run_cap_experiment(
    sql_cap: int,
    vector_cap: int,
    ranking_cap: int,
    dataset_size: int = 20_000,
) -> CapExperimentResult:
    """Run a single cap configuration experiment."""
    random.seed(42)
    products = generate_mock_products(dataset_size)
    query_emb = [round(random.gauss(0, 1), 4) for _ in range(128)]

    tracemalloc.start()

    # SQL filter
    t0 = time.perf_counter()
    sql_ids = simulate_sql_filter(products, sql_cap)
    sql_ms = (time.perf_counter() - t0) * 1000

    # Vector re-rank
    t1 = time.perf_counter()
    vector_ranked = simulate_vector_rerank(products, sql_ids, query_emb, vector_cap)
    vector_ms = (time.perf_counter() - t1) * 1000

    # Build ranking candidates
    candidates = build_ranking_candidates(products, vector_ranked, ranking_cap)

    # Ranking
    t2 = time.perf_counter()
    ranked = rank_candidates(candidates, budget_max=60_000)
    ranking_ms = (time.perf_counter() - t2) * 1000

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Determinism check: run ranking 10 times
    is_deterministic = True
    first_ids = [r["product_id"] for r in ranked]
    for _ in range(9):
        repeat = rank_candidates(candidates, budget_max=60_000)
        if [r["product_id"] for r in repeat] != first_ids:
            is_deterministic = False
            break

    total_ms = sql_ms + vector_ms + ranking_ms

    return CapExperimentResult(
        sql_cap=sql_cap,
        vector_cap=vector_cap,
        ranking_cap=ranking_cap,
        dataset_size=dataset_size,
        sql_filter_ms=round(sql_ms, 3),
        vector_rerank_ms=round(vector_ms, 3),
        ranking_ms=round(ranking_ms, 3),
        total_ms=round(total_ms, 3),
        sql_candidates=len(sql_ids),
        vector_candidates=len(vector_ranked),
        final_ranked=len(ranked),
        memory_peak_kb=round(peak / 1024, 2),
        ranking_deterministic=is_deterministic,
    )


def main() -> None:
    """Run candidate cap optimization experiment."""
    print("=" * 70)
    print("Evalio Candidate Cap Optimization Experiment")
    print("=" * 70)

    settings = get_settings()
    print(f"\nProduction defaults: SQL={settings.max_sql_candidates}, "
          f"Vector={settings.max_vector_candidates}, "
          f"Ranking={settings.max_rank_candidates}")
    print(f"(Production config NOT modified — using local overrides only)\n")

    # Test configurations
    configs = [
        # (sql_cap, vector_cap, ranking_cap)
        (300, 30, 20),   # Tighter
        (300, 50, 30),   # Current production
        (300, 75, 30),   # Wider vector
        (300, 100, 30),  # Wide vector
        (500, 50, 30),   # Override SQL cap (experiment only)
        (500, 75, 30),   # Override SQL + wider vector
        (500, 100, 30),  # Maximum
    ]

    results: list[CapExperimentResult] = []

    for sql_cap, vector_cap, ranking_cap in configs:
        result = run_cap_experiment(sql_cap, vector_cap, ranking_cap)
        results.append(result)

    # Print comparison table
    print(f"{'SQL Cap':>8} | {'Vec Cap':>8} | {'Rank Cap':>9} | "
          f"{'SQL ms':>8} | {'Vec ms':>8} | {'Rank ms':>8} | "
          f"{'Total ms':>9} | {'SQL→':>5} | {'Vec→':>5} | {'Final':>5} | "
          f"{'Mem KB':>8} | {'Determ.':>7}")
    print("-" * 120)

    for r in results:
        det_str = "✅" if r.ranking_deterministic else "❌"
        vec_warn = " ⚠️" if r.vector_rerank_ms > 120 else ""
        print(f"{r.sql_cap:>8} | {r.vector_cap:>8} | {r.ranking_cap:>9} | "
              f"{r.sql_filter_ms:>8.3f} | {r.vector_rerank_ms:>8.3f}{vec_warn:2s}| {r.ranking_ms:>8.3f} | "
              f"{r.total_ms:>9.3f} | {r.sql_candidates:>5} | {r.vector_candidates:>5} | {r.final_ranked:>5} | "
              f"{r.memory_peak_kb:>8.2f} | {det_str:>7}")

    # Find optimal
    valid = [r for r in results if r.vector_rerank_ms < 120 and r.ranking_deterministic]
    if valid:
        optimal = min(valid, key=lambda r: r.total_ms)
        print(f"\n🏆 Optimal config (under 120ms vector): "
              f"SQL={optimal.sql_cap}, Vector={optimal.vector_cap}, Ranking={optimal.ranking_cap} "
              f"— Total: {optimal.total_ms:.3f}ms")
    else:
        print("\n⚠️ No configuration met the 120ms vector latency target.")

    # All deterministic check
    all_det = all(r.ranking_deterministic for r in results)
    print(f"\n{'✅' if all_det else '❌'} Ranking determinism: {'ALL PASS' if all_det else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
