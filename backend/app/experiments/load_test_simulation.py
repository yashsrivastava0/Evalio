"""Load Test Simulation — sequential query simulation.

Simulates 100, 500, 1000 sequential queries through the mock retrieval
pipeline to measure latency distribution and cache behavior.

Usage:
    python -m app.experiments.load_test_simulation
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.services.ranking import rank_candidates

logger = get_logger(__name__)


def generate_products(count: int = 20_000) -> list[dict[str, Any]]:
    """Generate mock product dataset."""
    products = []
    for i in range(1, count + 1):
        products.append({
            "product_id": i,
            "category": random.choice(["smartphone", "laptop"]),
            "brand": random.choice(["samsung", "apple", "oneplus", "xiaomi", "dell", "hp"]),
            "price_inr": random.randint(8_000, 200_000),
            "ram_gb": random.choice([4, 6, 8, 12, 16]),
            "storage_gb": random.choice([64, 128, 256, 512]),
            "availability": random.random() > 0.1,
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


def generate_query() -> dict[str, Any]:
    """Generate a random query scenario."""
    budgets = [15_000, 30_000, 50_000, 80_000, 120_000, None]
    categories = ["smartphone", "laptop", None]
    rams = [None, 4, 6, 8, 12]

    return {
        "category": random.choice(categories),
        "budget_max": random.choice(budgets),
        "min_ram": random.choice(rams),
    }


def simulate_pipeline(
    products: list[dict],
    query: dict,
    cache: dict[str, Any],
) -> tuple[float, bool]:
    """Simulate full retrieval + ranking pipeline. Returns (latency_ms, cache_hit)."""
    import math

    # Check simple cache
    cache_key = f"{query.get('category')}:{query.get('budget_max')}:{query.get('min_ram')}"
    if cache_key in cache:
        return 0.5, True  # Cache hit — nearly instant

    t0 = time.perf_counter()

    # SQL filter
    filtered = products
    if query.get("category"):
        filtered = [p for p in filtered if p["category"] == query["category"]]
    if query.get("budget_max"):
        filtered = [p for p in filtered if p["price_inr"] <= query["budget_max"]]
    if query.get("min_ram"):
        filtered = [p for p in filtered if p["ram_gb"] >= query["min_ram"]]
    filtered = [p for p in filtered if p["availability"]]
    filtered = filtered[:300]

    # Vector re-rank (simplified)
    query_emb = [random.gauss(0, 1) for _ in range(128)]
    reranked = []
    for p in filtered[:100]:
        dot = sum(a * b for a, b in zip(query_emb, p["embedding"]))
        na = math.sqrt(sum(a * a for a in query_emb))
        nb = math.sqrt(sum(b * b for b in p["embedding"]))
        dist = 1.0 - (dot / (na * nb)) if na > 0 and nb > 0 else 1.0
        reranked.append((dist, p))
    reranked.sort(key=lambda x: x[0])
    top = reranked[:30]

    # Build ranking candidates
    candidates = []
    for dist, p in top:
        candidates.append({
            "product_id": p["product_id"],
            "product_slug": f"product-{p['product_id']}",
            "brand": p["brand"],
            "model_name": f"Model {p['product_id']}",
            "canonical_title": f"Product {p['product_id']}",
            "category": p["category"],
            "vector_distance": dist,
            "specs": {},
            "variants": [{"price_inr": p["price_inr"], "availability": True, "vendor": "test"}],
            "scores": p["scores"],
        })

    # Rank
    if candidates:
        rank_candidates(candidates, budget_max=query.get("budget_max"))

    latency = (time.perf_counter() - t0) * 1000

    # Store in cache
    cache[cache_key] = True

    return latency, False


@dataclass
class LoadTestResult:
    """Load test result."""
    query_count: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    cache_hit_rate: float
    total_time_s: float


def run_load_test(query_count: int, products: list[dict]) -> LoadTestResult:
    """Run sequential load test."""
    cache: dict[str, Any] = {}
    latencies: list[float] = []
    cache_hits = 0

    t_start = time.perf_counter()
    for _ in range(query_count):
        query = generate_query()
        latency, hit = simulate_pipeline(products, query, cache)
        latencies.append(latency)
        if hit:
            cache_hits += 1
    total_time = time.perf_counter() - t_start

    latencies.sort()
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = int(len(latencies) * 0.95)
    p99_idx = int(len(latencies) * 0.99)

    return LoadTestResult(
        query_count=query_count,
        mean_latency_ms=round(statistics.mean(latencies), 3),
        p50_latency_ms=round(latencies[p50_idx], 3),
        p95_latency_ms=round(latencies[p95_idx], 3),
        p99_latency_ms=round(latencies[p99_idx], 3),
        max_latency_ms=round(latencies[-1], 3),
        cache_hit_rate=round(cache_hits / query_count, 4),
        total_time_s=round(total_time, 3),
    )


def main() -> None:
    """Run load test simulations."""
    print("=" * 70)
    print("Evalio Load Test Simulation")
    print("=" * 70)

    random.seed(42)
    products = generate_products(20_000)
    print(f"Dataset: {len(products):,} products\n")

    query_counts = [100, 500, 1000]
    results: list[LoadTestResult] = []

    for count in query_counts:
        random.seed(42)  # Reset per run for reproducibility
        print(f"  Running {count:,} queries...", end="", flush=True)
        result = run_load_test(count, products)
        results.append(result)
        print(f" ✅ Mean: {result.mean_latency_ms:.3f}ms | "
              f"P95: {result.p95_latency_ms:.3f}ms | "
              f"Cache: {result.cache_hit_rate:.1%} | "
              f"Total: {result.total_time_s:.3f}s")

    print("\n--- Summary ---\n")
    print(f"{'Queries':>8} | {'Mean ms':>9} | {'P50 ms':>8} | {'P95 ms':>8} | "
          f"{'P99 ms':>8} | {'Max ms':>8} | {'Cache %':>8} | {'Total s':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r.query_count:>8} | {r.mean_latency_ms:>9.3f} | {r.p50_latency_ms:>8.3f} | "
              f"{r.p95_latency_ms:>8.3f} | {r.p99_latency_ms:>8.3f} | {r.max_latency_ms:>8.3f} | "
              f"{r.cache_hit_rate:>7.1%} | {r.total_time_s:>8.3f}")

    print("\n✅ Load simulation complete.")


if __name__ == "__main__":
    main()
