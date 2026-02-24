"""Retrieval Benchmark — Live DB Profiling.

Connects to Docker PostgreSQL and benchmarks actual retrieval performance
including real I/O, HNSW index usage, JOIN behavior, and cardinality effects.

Requires:
    - Docker PostgreSQL running with pgvector enabled
    - Product data seeded in the database

Usage:
    python -m app.experiments.retrieval_benchmark_live
"""

from __future__ import annotations

import asyncio
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.retrieval import sql_filter_candidates, vector_rerank_candidates, reduce_candidates

logger = get_logger(__name__)


@dataclass
class LiveBenchmarkResult:
    """Single live benchmark run result."""
    scenario: str
    sql_time_ms: float
    vector_time_ms: float
    reduce_time_ms: float
    candidate_count: int
    vector_candidates: int
    final_candidates: int
    total_retrieval_time_ms: float
    sql_plan_summary: str


LIVE_SCENARIOS: list[dict[str, Any]] = [
    {"name": "budget_under_30k", "category": "smartphone", "budget_max": 30_000},
    {"name": "budget_under_60k", "category": "smartphone", "budget_max": 60_000},
    {"name": "category_smartphone", "category": "smartphone"},
    {"name": "category_ram_8gb", "category": "smartphone", "min_ram": 8},
    {"name": "high_cardinality", "category": None},
    {"name": "brand_samsung_apple", "category": "smartphone", "brands": ["samsung", "apple"]},
    {"name": "tight_budget_15k", "category": "smartphone", "budget_max": 15_000},
]


async def get_explain_plan(db: AsyncSession, scenario: dict) -> str:
    """Run EXPLAIN ANALYZE on the SQL filter query for this scenario."""
    conditions = []
    params: dict[str, Any] = {}

    if scenario.get("category"):
        conditions.append("p.category = :category")
        params["category"] = scenario["category"]

    if scenario.get("budget_max"):
        conditions.append("v.price_inr <= :budget_max")
        params["budget_max"] = scenario["budget_max"]

    if scenario.get("brands"):
        conditions.append("p.brand = ANY(:brands)")
        params["brands"] = scenario["brands"]

    if scenario.get("min_ram"):
        conditions.append("v.ram_gb >= :min_ram")
        params["min_ram"] = scenario["min_ram"]

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    explain_sql = text(f"""
        EXPLAIN ANALYZE
        SELECT DISTINCT p.id
        FROM products p
        JOIN variants v ON v.product_id = p.id
        WHERE v.availability_bool = TRUE
          AND {where_clause}
        LIMIT 300
    """)

    try:
        result = await db.execute(explain_sql, params)
        rows = result.fetchall()
        return "\n".join(row[0] for row in rows)
    except Exception as e:
        return f"EXPLAIN failed: {e}"


async def run_live_scenario(
    db: AsyncSession,
    scenario: dict,
    query_embedding: list[float],
) -> LiveBenchmarkResult:
    """Run a single benchmark scenario against live DB."""
    name = scenario["name"]
    logger.info("live_benchmark_start", scenario=name)

    # Get EXPLAIN plan
    plan = await get_explain_plan(db, scenario)

    # Step 1: SQL Filter (timed)
    t0 = time.perf_counter()
    filtered_ids = await sql_filter_candidates(
        db,
        category=scenario.get("category"),
        budget_max=scenario.get("budget_max"),
        brands=scenario.get("brands"),
        min_ram=scenario.get("min_ram"),
    )
    sql_time = (time.perf_counter() - t0) * 1000

    # Step 2: Vector Re-rank (timed)
    t1 = time.perf_counter()
    vector_ranked = await vector_rerank_candidates(
        db,
        filtered_ids,
        query_embedding,
    )
    vector_time = (time.perf_counter() - t1) * 1000

    # Step 3: Candidate Reduction (timed)
    t2 = time.perf_counter()
    final = await reduce_candidates(db, vector_ranked)
    reduce_time = (time.perf_counter() - t2) * 1000

    total_time = sql_time + vector_time + reduce_time

    # Summarize plan
    plan_summary = "HNSW" if "hnsw" in plan.lower() else "Index Scan" if "index" in plan.lower() else "Seq Scan ⚠️" if "seq scan" in plan.lower() else "Unknown"

    result = LiveBenchmarkResult(
        scenario=name,
        sql_time_ms=round(sql_time, 3),
        vector_time_ms=round(vector_time, 3),
        reduce_time_ms=round(reduce_time, 3),
        candidate_count=len(filtered_ids),
        vector_candidates=len(vector_ranked),
        final_candidates=len(final),
        total_retrieval_time_ms=round(total_time, 3),
        sql_plan_summary=plan_summary,
    )

    logger.info(
        "live_benchmark_done",
        scenario=name,
        sql_ms=result.sql_time_ms,
        vector_ms=result.vector_time_ms,
        candidates=result.candidate_count,
        final=result.final_candidates,
        plan=plan_summary,
    )

    return result


async def run_all_live_benchmarks() -> list[LiveBenchmarkResult]:
    """Run all live benchmark scenarios."""
    settings = get_settings()

    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Generate a random query embedding (consistent seed)
    import random
    random.seed(42)
    dimension = settings.embedding_dimension
    query_embedding = [round(random.gauss(0, 1), 6) for _ in range(dimension)]

    results: list[LiveBenchmarkResult] = []

    async with async_session() as db:
        # Check product count first
        count_result = await db.execute(text("SELECT COUNT(*) FROM products"))
        product_count = count_result.scalar()
        print(f"📊 Products in database: {product_count}")

        if product_count == 0:
            print("⚠️  No products in database. Seed data first.")
            print("   Live benchmarking requires product data in PostgreSQL.")
            await engine.dispose()
            return []

        for scenario in LIVE_SCENARIOS:
            try:
                result = await run_live_scenario(db, scenario, query_embedding)
                results.append(result)
            except Exception as e:
                logger.error("live_benchmark_error", scenario=scenario["name"], error=str(e))
                print(f"  ❌ {scenario['name']}: {e}")

    await engine.dispose()
    return results


def write_live_csv(results: list[LiveBenchmarkResult], output_path: str = "retrieval_benchmark_live_results.csv") -> None:
    """Write live benchmark results to CSV."""
    path = Path(output_path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "sql_time_ms", "vector_time_ms", "reduce_time_ms",
            "candidate_count", "vector_candidates", "final_candidates",
            "total_retrieval_time_ms", "sql_plan_summary",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scenario": r.scenario,
                "sql_time_ms": r.sql_time_ms,
                "vector_time_ms": r.vector_time_ms,
                "reduce_time_ms": r.reduce_time_ms,
                "candidate_count": r.candidate_count,
                "vector_candidates": r.vector_candidates,
                "final_candidates": r.final_candidates,
                "total_retrieval_time_ms": r.total_retrieval_time_ms,
                "sql_plan_summary": r.sql_plan_summary,
            })
    logger.info("live_benchmark_csv_written", path=str(path))


def main() -> None:
    """Run live benchmarks."""
    print("=" * 60)
    print("Evalio Retrieval Benchmark — Live DB Profiling")
    print("=" * 60)

    results = asyncio.run(run_all_live_benchmarks())

    if not results:
        print("\n❌ No results — database may be empty.")
        return

    print("\n--- Live Benchmark Results ---")
    for r in results:
        print(f"  {r.scenario:25s} | SQL: {r.sql_time_ms:8.3f}ms | "
              f"Vector: {r.vector_time_ms:8.3f}ms | "
              f"Reduce: {r.reduce_time_ms:8.3f}ms | "
              f"Candidates: {r.candidate_count:5d} → {r.vector_candidates:3d} → {r.final_candidates:3d} | "
              f"Total: {r.total_retrieval_time_ms:8.3f}ms | Plan: {r.sql_plan_summary}")

    write_live_csv(results)
    print(f"\n✅ Results written to retrieval_benchmark_live_results.csv")


if __name__ == "__main__":
    main()
