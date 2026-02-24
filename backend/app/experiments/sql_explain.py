"""SQL Explain Analysis Tool — EXPLAIN ANALYZE for retrieval queries.

Runs EXPLAIN ANALYZE on SQL filter and vector JOIN queries against live DB.
Detects: Seq Scan presence, index usage, HNSW index usage.
Raises warning on full table scan detection.

Requires live Docker PostgreSQL.

Usage:
    python -m app.experiments.sql_explain
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── EXPLAIN Queries ──

SQL_FILTER_EXPLAIN = """
EXPLAIN ANALYZE
SELECT DISTINCT p.id
FROM products p
JOIN variants v ON v.product_id = p.id
WHERE v.availability_bool = TRUE
  AND p.category = :category
  AND v.price_inr <= :budget_max
LIMIT 300
"""

VECTOR_JOIN_EXPLAIN = """
EXPLAIN ANALYZE
SELECT e.product_id,
       e.embedding_vector <=> :query_vec AS distance
FROM embeddings e
WHERE e.product_id = ANY(:product_ids)
  AND e.embedding_type = :emb_type
ORDER BY e.embedding_vector <=> :query_vec
LIMIT 50
"""


# ── Plan Analysis ──

def analyze_plan(plan_lines: list[str]) -> dict[str, Any]:
    """Analyze an EXPLAIN ANALYZE output for scan types and index usage."""
    full_plan = "\n".join(plan_lines)
    plan_lower = full_plan.lower()

    analysis = {
        "has_seq_scan": "seq scan" in plan_lower,
        "has_index_scan": "index scan" in plan_lower or "index only scan" in plan_lower,
        "has_bitmap_scan": "bitmap" in plan_lower,
        "has_hnsw": "hnsw" in plan_lower,
        "has_ivfflat": "ivfflat" in plan_lower,
        "execution_time_ms": None,
        "planning_time_ms": None,
        "rows_returned": None,
        "warnings": [],
    }

    for line in plan_lines:
        line_lower = line.lower().strip()
        if "execution time:" in line_lower:
            try:
                analysis["execution_time_ms"] = float(line_lower.split("execution time:")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                pass
        if "planning time:" in line_lower:
            try:
                analysis["planning_time_ms"] = float(line_lower.split("planning time:")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                pass
        if "rows=" in line_lower:
            try:
                parts = line_lower.split("rows=")
                if len(parts) > 1:
                    rows = int(parts[-1].split(" ")[0].split(")")[0])
                    if analysis["rows_returned"] is None or rows > analysis["rows_returned"]:
                        analysis["rows_returned"] = rows
            except (ValueError, IndexError):
                pass

    # Warnings
    if analysis["has_seq_scan"]:
        analysis["warnings"].append("⚠️  FULL TABLE SEQ SCAN DETECTED — add index or filter narrowing")

    if not analysis["has_index_scan"] and not analysis["has_bitmap_scan"]:
        analysis["warnings"].append("⚠️  No index scan detected — query may not be using indexes")

    return analysis


async def run_sql_filter_explain(db: AsyncSession, category: str, budget_max: float) -> dict[str, Any]:
    """Run EXPLAIN ANALYZE on SQL filter query."""
    result = await db.execute(
        text(SQL_FILTER_EXPLAIN),
        {"category": category, "budget_max": budget_max},
    )
    plan_lines = [row[0] for row in result.fetchall()]
    analysis = analyze_plan(plan_lines)
    return {"plan": plan_lines, "analysis": analysis}


async def run_vector_join_explain(
    db: AsyncSession,
    product_ids: list[int],
    query_vec: str,
    emb_type: str = "specs",
) -> dict[str, Any]:
    """Run EXPLAIN ANALYZE on vector JOIN query."""
    if not product_ids:
        return {"plan": ["No product IDs to test."], "analysis": {"warnings": ["No data"]}}

    result = await db.execute(
        text(VECTOR_JOIN_EXPLAIN),
        {
            "product_ids": product_ids,
            "query_vec": query_vec,
            "emb_type": emb_type,
        },
    )
    plan_lines = [row[0] for row in result.fetchall()]
    analysis = analyze_plan(plan_lines)
    return {"plan": plan_lines, "analysis": analysis}


async def run_full_analysis() -> None:
    """Run full SQL explain analysis against live DB."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Check product count
        count_result = await db.execute(text("SELECT COUNT(*) FROM products"))
        product_count = count_result.scalar()
        print(f"📊 Products in database: {product_count}")

        if product_count == 0:
            print("⚠️  No products in database. Cannot run EXPLAIN ANALYZE.")
            await engine.dispose()
            return

        # 1. SQL Filter Explain
        print("\n" + "=" * 60)
        print("1. SQL FILTER QUERY — EXPLAIN ANALYZE")
        print("=" * 60)

        sql_result = await run_sql_filter_explain(db, "smartphone", 60_000)
        print("\nExecution Plan:")
        for line in sql_result["plan"]:
            print(f"  {line}")

        analysis = sql_result["analysis"]
        print(f"\n  Index Scan: {'✅' if analysis['has_index_scan'] else '❌'}")
        print(f"  Seq Scan:   {'⚠️  YES' if analysis['has_seq_scan'] else '✅ No'}")
        print(f"  Exec Time:  {analysis['execution_time_ms']}ms")
        for w in analysis["warnings"]:
            print(f"  {w}")

        # Get some product IDs for vector explain
        id_result = await db.execute(text("SELECT id FROM products LIMIT 50"))
        product_ids = [row[0] for row in id_result.fetchall()]

        # 2. Vector JOIN Explain
        print("\n" + "=" * 60)
        print("2. VECTOR JOIN QUERY — EXPLAIN ANALYZE")
        print("=" * 60)

        # Generate fake query vector
        import random
        random.seed(42)
        dim = settings.embedding_dimension
        vec_str = "[" + ",".join(str(round(random.gauss(0, 1), 6)) for _ in range(dim)) + "]"

        vec_result = await run_vector_join_explain(db, product_ids, vec_str)
        print("\nExecution Plan:")
        for line in vec_result["plan"]:
            print(f"  {line}")

        vec_analysis = vec_result["analysis"]
        print(f"\n  HNSW:       {'✅' if vec_analysis['has_hnsw'] else '❌ Not detected'}")
        print(f"  Index Scan: {'✅' if vec_analysis['has_index_scan'] else '❌'}")
        print(f"  Seq Scan:   {'⚠️  YES' if vec_analysis['has_seq_scan'] else '✅ No'}")
        print(f"  Exec Time:  {vec_analysis['execution_time_ms']}ms")
        for w in vec_analysis["warnings"]:
            print(f"  {w}")

    await engine.dispose()

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    all_warnings = sql_result["analysis"]["warnings"] + vec_result["analysis"]["warnings"]
    if not all_warnings:
        print("  ✅ All queries use indexes. No full table scans detected.")
    else:
        print("  ⚠️  Issues found:")
        for w in all_warnings:
            print(f"    {w}")


def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("Evalio SQL Explain Analysis Tool")
    print("=" * 60)
    asyncio.run(run_full_analysis())


if __name__ == "__main__":
    main()
