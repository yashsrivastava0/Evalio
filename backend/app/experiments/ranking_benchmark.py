"""Ranking Performance Benchmark.

Measures ranking latency for various candidate counts.
Asserts <20ms for ≤50 candidates.
Verifies 100-run determinism across all sizes.

Usage:
    python -m app.experiments.ranking_benchmark
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.services.ranking import rank_candidates

logger = get_logger(__name__)


def generate_ranking_candidates(count: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate deterministic ranking candidates."""
    random.seed(seed)

    candidates = []
    for i in range(1, count + 1):
        candidates.append({
            "product_id": i,
            "product_slug": f"product-{i}",
            "brand": random.choice(["samsung", "apple", "oneplus", "xiaomi", "nothing"]),
            "model_name": f"Model {i}",
            "canonical_title": f"Test Product {i}",
            "category": "smartphone",
            "vector_distance": round(random.uniform(0.01, 0.99), 4),
            "specs": {},
            "variants": [
                {
                    "price_inr": random.randint(10_000, 80_000),
                    "availability": True,
                    "vendor": "test",
                }
            ],
            "scores": {
                "performance_score": round(random.uniform(0.1, 1.0), 4),
                "battery_score": round(random.uniform(0.1, 1.0), 4),
                "camera_score": round(random.uniform(0.1, 1.0), 4),
                "display_score": round(random.uniform(0.1, 1.0), 4),
                "build_score": round(random.uniform(0.1, 1.0), 4),
                "price_score": round(random.uniform(0.1, 1.0), 4),
                "popularity_score": round(random.uniform(0.0, 1.0), 4),
            },
        })

    return candidates


@dataclass
class RankingBenchmarkResult:
    """Single ranking benchmark result."""
    candidate_count: int
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    is_deterministic: bool
    meets_target: bool  # < 20ms for <= 50 candidates


def benchmark_ranking(candidate_count: int, runs: int = 100) -> RankingBenchmarkResult:
    """Benchmark ranking for a specific candidate count."""
    candidates = generate_ranking_candidates(candidate_count)

    latencies: list[float] = []
    first_result_ids: list[int] | None = None
    is_deterministic = True

    for i in range(runs):
        t0 = time.perf_counter()
        ranked = rank_candidates(candidates, budget_max=60_000)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        current_ids = [r["product_id"] for r in ranked]
        if first_result_ids is None:
            first_result_ids = current_ids
        elif current_ids != first_result_ids:
            is_deterministic = False

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    p95_idx = int(len(latencies) * 0.95)

    # Target: < 20ms for <= 50 candidates
    target = 20.0 if candidate_count <= 50 else float("inf")

    return RankingBenchmarkResult(
        candidate_count=candidate_count,
        mean_latency_ms=round(mean_ms, 3),
        min_latency_ms=round(latencies[0], 3),
        max_latency_ms=round(latencies[-1], 3),
        p95_latency_ms=round(latencies[p95_idx], 3),
        is_deterministic=is_deterministic,
        meets_target=mean_ms < target,
    )


def main() -> None:
    """Run ranking benchmark across all candidate sizes."""
    print("=" * 70)
    print("Evalio Ranking Performance Benchmark")
    print("=" * 70)
    print(f"Runs per size: 100")
    print(f"Target: < 20ms for ≤ 50 candidates\n")

    candidate_sizes = [10, 20, 50, 100]
    results: list[RankingBenchmarkResult] = []

    for size in candidate_sizes:
        print(f"  Benchmarking {size} candidates...", end="", flush=True)
        result = benchmark_ranking(size, runs=100)
        results.append(result)
        status = "✅" if result.meets_target and result.is_deterministic else "❌"
        print(f" {status} Mean: {result.mean_latency_ms:.3f}ms | "
              f"P95: {result.p95_latency_ms:.3f}ms | "
              f"Deterministic: {'Yes' if result.is_deterministic else 'NO'}")

    print("\n--- Summary ---\n")
    print(f"{'Candidates':>12} | {'Mean ms':>9} | {'Min ms':>8} | {'Max ms':>8} | "
          f"{'P95 ms':>8} | {'Determ.':>8} | {'Target':>7}")
    print("-" * 80)

    for r in results:
        det = "✅" if r.is_deterministic else "❌"
        tgt = "✅ PASS" if r.meets_target else "❌ FAIL" if r.candidate_count <= 50 else "N/A"
        print(f"{r.candidate_count:>12} | {r.mean_latency_ms:>9.3f} | {r.min_latency_ms:>8.3f} | "
              f"{r.max_latency_ms:>8.3f} | {r.p95_latency_ms:>8.3f} | {det:>8} | {tgt:>7}")

    # Final assertions
    all_deterministic = all(r.is_deterministic for r in results)
    all_targets_met = all(r.meets_target for r in results if r.candidate_count <= 50)

    print(f"\n{'✅' if all_deterministic else '❌'} Determinism: "
          f"{'ALL 100-run checks passed' if all_deterministic else 'FAILURES DETECTED'}")
    print(f"{'✅' if all_targets_met else '❌'} Latency targets: "
          f"{'ALL PASS (< 20ms for ≤ 50 candidates)' if all_targets_met else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
