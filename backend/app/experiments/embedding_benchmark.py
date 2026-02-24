"""Embedding Benchmark Harness — offline MAP@10 + Recall@10 measurement.

Offline-only: Uses mock vectors. No DB required. No external API calls.

Usage:
    cd backend
    python -m app.experiments.embedding_benchmark

Reports:
    - MAP@10  (Mean Average Precision at 10)
    - Recall@10
    - Avg retrieval latency (microseconds)
    - Candidate reduction efficiency

Tests both specs-only and reviews-only embedding paths against
a synthetic query set with ground-truth product labels.
"""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Ground-Truth Dataset ──────────────────────────────────────────────────────

@dataclass
class SyntheticQuery:
    query_id: str
    query_text: str
    relevant_product_ids: list[int]
    embedding_type: str  # "specs" or "review"


def _build_synthetic_queries(n_products: int = 50, n_queries: int = 300) -> list[SyntheticQuery]:
    """Generate synthetic query set with deterministic relevance labels.

    Each query has 3-5 "relevant" products from the mock catalog.
    Uses seeded random for reproducibility.
    """
    rng = random.Random(42)
    query_templates = [
        ("best phone under 30000", "specs"),
        ("phone with great camera", "review"),
        ("long battery life phone", "review"),
        ("best performance phone 2024", "specs"),
        ("budget smartphone 15000", "specs"),
        ("phone for gaming", "specs"),
        ("phone with heating issues", "review"),
        ("best build quality phone", "review"),
        ("fast charging phone India", "specs"),
        ("best camera under 20k", "review"),
        ("phone with 5G support", "specs"),
        ("smartphone 8GB RAM 256GB", "specs"),
        ("phone with OLED display", "specs"),
        ("waterproof phone", "review"),
        ("phone for battery backup", "review"),
    ]

    queries = []
    for i in range(n_queries):
        tmpl = query_templates[i % len(query_templates)]
        # Each query is "relevant" to 3-5 random products
        n_relevant = rng.randint(3, 5)
        relevant = rng.sample(range(1, n_products + 1), n_relevant)
        queries.append(SyntheticQuery(
            query_id=f"q{i:04d}",
            query_text=tmpl[0],
            relevant_product_ids=relevant,
            embedding_type=tmpl[1],
        ))

    return queries


# ── Mock Retrieval ────────────────────────────────────────────────────────────

def _mock_vector(seed: int, dim: int = 1536) -> list[float]:
    """Generate a deterministic unit vector for a product."""
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def mock_retrieve(
    query_embedding: list[float],
    product_embeddings: dict[int, list[float]],
    k: int = 10,
) -> list[int]:
    """Retrieve top-k products by cosine distance.

    Simulates the HNSW vector re-rank step in isolation.
    Returns product_ids ordered by similarity.
    """
    distances = [
        (pid, _cosine_distance(query_embedding, emb))
        for pid, emb in product_embeddings.items()
    ]
    distances.sort(key=lambda x: x[1])
    return [pid for pid, _ in distances[:k]]


# ── Metrics ───────────────────────────────────────────────────────────────────

def average_precision_at_k(retrieved: list[int], relevant: list[int], k: int = 10) -> float:
    """Compute Average Precision@K for a single query."""
    relevant_set = set(relevant)
    hits = 0
    sum_precision = 0.0
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    if not relevant:
        return 0.0
    return sum_precision / min(len(relevant), k)


def recall_at_k(retrieved: list[int], relevant: list[int], k: int = 10) -> float:
    """Compute Recall@K for a single query."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


# ── Benchmark Runner ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    method: str
    map_at_10: float
    recall_at_10: float
    avg_latency_us: float
    n_queries: int
    candidate_reduction_pct: float


def run_benchmark(
    queries: list[SyntheticQuery],
    specs_embeddings: dict[int, list[float]],
    reviews_embeddings: dict[int, list[float]],
    k: int = 10,
) -> list[BenchmarkResult]:
    """Run benchmark for:
        1. specs-only embedding
        2. reviews-only embedding
    """
    results = []

    for method, emb_map, filter_fn in [
        ("specs_only", specs_embeddings, lambda q: q.embedding_type == "specs"),
        ("reviews_only", reviews_embeddings, lambda q: q.embedding_type == "review"),
        ("all_queries_specs", specs_embeddings, lambda q: True),
    ]:
        method_queries = [q for q in queries if filter_fn(q)]
        if not method_queries:
            continue

        map_scores = []
        recall_scores = []
        latencies_us = []
        total_candidates = len(specs_embeddings)

        for query in method_queries:
            # Use query text hash as deterministic query embedding
            q_seed = sum(ord(c) for c in query.query_text)
            query_emb = _mock_vector(q_seed)

            t0 = time.perf_counter()
            retrieved = mock_retrieve(query_emb, emb_map, k=k)
            elapsed_us = (time.perf_counter() - t0) * 1_000_000

            ap = average_precision_at_k(retrieved, query.relevant_product_ids, k)
            rec = recall_at_k(retrieved, query.relevant_product_ids, k)

            map_scores.append(ap)
            recall_scores.append(rec)
            latencies_us.append(elapsed_us)

        candidate_reduction_pct = (1 - k / total_candidates) * 100 if total_candidates > 0 else 0

        results.append(BenchmarkResult(
            method=method,
            map_at_10=round(sum(map_scores) / len(map_scores), 4),
            recall_at_10=round(sum(recall_scores) / len(recall_scores), 4),
            avg_latency_us=round(sum(latencies_us) / len(latencies_us), 2),
            n_queries=len(method_queries),
            candidate_reduction_pct=round(candidate_reduction_pct, 1),
        ))

    return results


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Evalio Phase 6 — Embedding Benchmark Harness")
    print("=" * 65)
    print(f"Generating synthetic queries...")

    n_products = 100
    queries = _build_synthetic_queries(n_products=n_products, n_queries=300)
    print(f"  Queries: {len(queries)}")

    # Generate mock product embeddings (deterministic, seeded)
    print(f"Generating {n_products} mock product embeddings (specs + reviews)...")
    specs_embeddings = {pid: _mock_vector(pid * 2) for pid in range(1, n_products + 1)}
    reviews_embeddings = {pid: _mock_vector(pid * 3 + 1) for pid in range(1, n_products + 1)}

    print("Running benchmark...")
    results = run_benchmark(queries, specs_embeddings, reviews_embeddings, k=10)

    # Print results table
    print()
    print(f"{'Method':<22} {'MAP@10':>8} {'Recall@10':>10} {'Avg Lat (µs)':>13} {'Queries':>8} {'Reduction':>10}")
    print("-" * 75)
    for r in results:
        print(
            f"{r.method:<22} {r.map_at_10:>8.4f} {r.recall_at_10:>10.4f} "
            f"{r.avg_latency_us:>13.2f} {r.n_queries:>8} {r.candidate_reduction_pct:>9.1f}%"
        )

    # Write CSV
    output_path = Path(__file__).parent / "benchmark_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "map_at_10", "recall_at_10", "avg_latency_us", "n_queries", "candidate_reduction_pct"])
        for r in results:
            writer.writerow([r.method, r.map_at_10, r.recall_at_10, r.avg_latency_us, r.n_queries, r.candidate_reduction_pct])

    print()
    print(f"Results saved to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
