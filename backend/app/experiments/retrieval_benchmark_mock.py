"""Retrieval Benchmark — Mock/Algorithm Scaling Test.

Simulates 10k–50k product datasets and benchmarks the retrieval pipeline
algorithm performance (SQL filter, vector re-rank, candidate reduction)
without requiring a live database.

Usage:
    python -m app.experiments.retrieval_benchmark_mock

Output:
    retrieval_benchmark_results.csv
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Synthetic Data Generator ──


def generate_synthetic_products(count: int = 10_000) -> list[dict[str, Any]]:
    """Generate synthetic product dataset with realistic distributions.

    Each product has: id, category, brand, price, ram_gb, storage_gb,
    availability, component scores, and a fake embedding vector (128-d).
    """
    categories = ["smartphone", "laptop"]
    brands_smartphone = ["samsung", "apple", "oneplus", "xiaomi", "nothing", "google", "motorola", "vivo", "oppo", "realme"]
    brands_laptop = ["dell", "hp", "lenovo", "asus", "apple", "acer", "msi"]

    products = []
    for i in range(1, count + 1):
        cat = random.choice(categories)
        brand = random.choice(brands_smartphone if cat == "smartphone" else brands_laptop)
        price = random.randint(8_000, 200_000) if cat == "smartphone" else random.randint(25_000, 300_000)
        ram = random.choice([4, 6, 8, 12, 16]) if cat == "smartphone" else random.choice([8, 16, 32, 64])
        storage = random.choice([64, 128, 256, 512]) if cat == "smartphone" else random.choice([256, 512, 1024, 2048])

        products.append({
            "product_id": i,
            "category": cat,
            "brand": brand,
            "price_inr": price,
            "ram_gb": ram,
            "storage_gb": storage,
            "availability": random.random() > 0.1,  # 90% available
            "scores": {
                "performance_score": round(random.uniform(0.1, 1.0), 4),
                "battery_score": round(random.uniform(0.1, 1.0), 4),
                "camera_score": round(random.uniform(0.1, 1.0), 4),
                "display_score": round(random.uniform(0.1, 1.0), 4),
                "build_score": round(random.uniform(0.1, 1.0), 4),
                "price_score": round(random.uniform(0.1, 1.0), 4),
                "popularity_score": round(random.uniform(0.0, 1.0), 4),
            },
            # Fake 128-d embedding for algorithm scaling test
            "embedding": [round(random.gauss(0, 1), 6) for _ in range(128)],
        })

    return products


# ── Mock Retrieval Pipeline ──


def mock_sql_filter(
    products: list[dict],
    *,
    category: str | None = None,
    budget_max: float | None = None,
    brands: list[str] | None = None,
    min_ram: int | None = None,
    availability_only: bool = True,
    max_candidates: int = 300,
) -> list[int]:
    """Simulate SQL filter on synthetic dataset."""
    filtered = products

    if category:
        filtered = [p for p in filtered if p["category"] == category]
    if availability_only:
        filtered = [p for p in filtered if p["availability"]]
    if budget_max is not None:
        filtered = [p for p in filtered if p["price_inr"] <= budget_max]
    if brands:
        brands_lower = [b.lower() for b in brands]
        filtered = [p for p in filtered if p["brand"] in brands_lower]
    if min_ram is not None:
        filtered = [p for p in filtered if p["ram_gb"] >= min_ram]

    return [p["product_id"] for p in filtered[:max_candidates]]


def mock_vector_rerank(
    products: list[dict],
    filtered_ids: list[int],
    query_embedding: list[float],
    *,
    max_candidates: int = 50,
) -> list[dict]:
    """Simulate vector re-rank via cosine distance on filtered subset."""
    import math

    def cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

    # Build lookup for fast access
    id_set = set(filtered_ids)
    subset = [p for p in products if p["product_id"] in id_set]

    # Compute distances (only on filtered subset — no full scan)
    ranked = []
    for p in subset:
        dist = cosine_distance(query_embedding, p["embedding"])
        ranked.append({"product_id": p["product_id"], "distance": dist})

    ranked.sort(key=lambda x: x["distance"])
    return ranked[:max_candidates]


def mock_reduce_candidates(
    products: list[dict],
    vector_ranked: list[dict],
    *,
    max_candidates: int = 30,
) -> list[dict]:
    """Simulate candidate reduction — fetch full data for top N."""
    top_ids = set(c["product_id"] for c in vector_ranked[:max_candidates])
    product_map = {p["product_id"]: p for p in products}

    result = []
    for c in vector_ranked[:max_candidates]:
        pid = c["product_id"]
        if pid in product_map:
            p = product_map[pid]
            result.append({
                "product_id": pid,
                "brand": p["brand"],
                "category": p["category"],
                "price_inr": p["price_inr"],
                "vector_distance": c["distance"],
                "scores": p["scores"],
            })
    return result


# ── Benchmark Scenarios ──

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    scenario: str
    dataset_size: int
    sql_time_ms: float
    vector_time_ms: float
    candidate_count: int
    final_candidates: int
    total_retrieval_time_ms: float


SCENARIOS: list[dict[str, Any]] = [
    {"name": "budget_under_30k", "category": "smartphone", "budget_max": 30_000},
    {"name": "budget_under_60k", "category": "smartphone", "budget_max": 60_000},
    {"name": "category_smartphone", "category": "smartphone"},
    {"name": "category_ram_filter", "category": "smartphone", "min_ram": 8},
    {"name": "high_cardinality", "category": None},  # No filter — tests worst case
    {"name": "brand_filter", "category": "smartphone", "brands": ["samsung", "apple"]},
    {"name": "tight_budget", "category": "smartphone", "budget_max": 15_000},
]


def run_benchmark(dataset_size: int = 10_000) -> list[BenchmarkResult]:
    """Run all benchmark scenarios on synthetic dataset."""
    random.seed(42)  # Deterministic generation
    products = generate_synthetic_products(dataset_size)
    query_embedding = [round(random.gauss(0, 1), 6) for _ in range(128)]

    results: list[BenchmarkResult] = []

    for scenario in SCENARIOS:
        name = scenario["name"]
        logger.info("benchmark_scenario_start", scenario=name, dataset_size=dataset_size)

        # SQL Filter
        t0 = time.perf_counter()
        filtered_ids = mock_sql_filter(
            products,
            category=scenario.get("category"),
            budget_max=scenario.get("budget_max"),
            brands=scenario.get("brands"),
            min_ram=scenario.get("min_ram"),
        )
        sql_time = (time.perf_counter() - t0) * 1000

        # Vector Re-rank
        t1 = time.perf_counter()
        vector_ranked = mock_vector_rerank(products, filtered_ids, query_embedding)
        vector_time = (time.perf_counter() - t1) * 1000

        # Candidate Reduction
        t2 = time.perf_counter()
        final = mock_reduce_candidates(products, vector_ranked)
        reduce_time = (time.perf_counter() - t2) * 1000

        total_time = sql_time + vector_time + reduce_time

        result = BenchmarkResult(
            scenario=name,
            dataset_size=dataset_size,
            sql_time_ms=round(sql_time, 3),
            vector_time_ms=round(vector_time, 3),
            candidate_count=len(filtered_ids),
            final_candidates=len(final),
            total_retrieval_time_ms=round(total_time, 3),
        )
        results.append(result)

        logger.info(
            "benchmark_scenario_done",
            scenario=name,
            sql_ms=result.sql_time_ms,
            vector_ms=result.vector_time_ms,
            candidates=result.candidate_count,
            final=result.final_candidates,
            total_ms=result.total_retrieval_time_ms,
        )

    return results


def write_csv(results: list[BenchmarkResult], output_path: str = "retrieval_benchmark_results.csv") -> None:
    """Write benchmark results to CSV."""
    path = Path(output_path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "dataset_size", "sql_time_ms", "vector_time_ms",
            "candidate_count", "final_candidates", "total_retrieval_time_ms",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scenario": r.scenario,
                "dataset_size": r.dataset_size,
                "sql_time_ms": r.sql_time_ms,
                "vector_time_ms": r.vector_time_ms,
                "candidate_count": r.candidate_count,
                "final_candidates": r.final_candidates,
                "total_retrieval_time_ms": r.total_retrieval_time_ms,
            })
    logger.info("benchmark_csv_written", path=str(path))


# ── Main ──

def main() -> None:
    """Run benchmarks across multiple dataset sizes."""
    print("=" * 60)
    print("Evalio Retrieval Benchmark — Mock/Algorithm Scaling")
    print("=" * 60)

    all_results: list[BenchmarkResult] = []

    for size in [10_000, 25_000, 50_000]:
        print(f"\n--- Dataset size: {size:,} products ---")
        results = run_benchmark(dataset_size=size)
        all_results.extend(results)

        for r in results:
            print(f"  {r.scenario:25s} | SQL: {r.sql_time_ms:8.3f}ms | "
                  f"Vector: {r.vector_time_ms:8.3f}ms | "
                  f"Candidates: {r.candidate_count:5d} → {r.final_candidates:3d} | "
                  f"Total: {r.total_retrieval_time_ms:8.3f}ms")

    write_csv(all_results)
    print(f"\n✅ Results written to retrieval_benchmark_results.csv")


if __name__ == "__main__":
    main()
