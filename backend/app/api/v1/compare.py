"""Compare API router — wraps the unified chat graph.

POST /v1/compare → CompareRequest → LangGraph → CompareResponse
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_embedding_provider, get_llm_client, get_redis
from app.graph import compile_graph
from app.schemas.compare import (
    CompareRequest,
    CompareResponse,
    ComparisonTableRow,
    ProCon,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/compare", tags=["compare"])


@router.post("", response_model=CompareResponse)
async def compare(
    request: CompareRequest,
    redis: object = Depends(get_redis),
    db: AsyncSession = Depends(get_db),
    llm_client: object = Depends(get_llm_client),
    embedding_provider: object = Depends(get_embedding_provider),
) -> CompareResponse:
    """Compare 2–4 products via the AI Product Decision Engine.

    Builds a natural language compare query from the provided slugs and
    delegates to the unified graph. Maps comparison_table component output
    into the CompareResponse format.
    """
    logger.info("compare_request", slugs=request.product_slugs, aspects=request.aspects)

    if len(request.product_slugs) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 products to compare")

    session_id = request.session_id or str(uuid4())

    # Build a natural-language compare query from the slugs
    slugs_str = " vs ".join(request.product_slugs)
    aspects_str = f" focusing on {', '.join(request.aspects)}" if request.aspects else ""
    query = f"compare {slugs_str}{aspects_str}"

    try:
        graph = compile_graph(
            redis_client=redis,
            db_session=db,
            llm_client=llm_client,
            embedding_provider=embedding_provider,
        )
        result = await graph.ainvoke({
            "query": query,
            "session_id": session_id,
        })

        structured = result.get("structured_output") or {}
        components = structured.get("components", [])

        # Extract comparison_table component
        comparison_table: list[ComparisonTableRow] = []
        pros_cons: list[ProCon] = []
        verdict = ""
        ranked_list = list(request.product_slugs)

        for component in components:
            if component.get("component") != "comparison_table":
                continue

            # Map technical_diffs → ComparisonTableRow
            for diff in component.get("technical_diffs", []):
                comparison_table.append(ComparisonTableRow(
                    aspect=diff.get("attribute", ""),
                    values={k: float(v) if isinstance(v, (int, float)) else 0.0
                            for k, v in diff.get("values", {}).items()},
                    winner=diff.get("winner") or "",
                ))

            # Map per-product pros/cons
            for p in component.get("products", []):
                slug = p.get("slug") or p.get("id") or ""
                if slug:
                    pros_cons.append(ProCon(
                        product_slug=str(slug),
                        pros=p.get("pros", []),
                        cons=p.get("cons", []),
                    ))

            verdict = component.get("overall_recommendation", "")

            # Respect per_category_winners for ordering if available
            winners = component.get("per_category_winners", {})
            if winners:
                winner_counts: dict[str, int] = {}
                for w in winners.values():
                    winner_counts[w] = winner_counts.get(w, 0) + 1
                ranked_list = sorted(
                    request.product_slugs,
                    key=lambda s: winner_counts.get(s, 0),
                    reverse=True,
                )
            break

        return CompareResponse(
            ranked_list=ranked_list,
            comparison_table=comparison_table,
            numeric_diffs=[],
            pros_cons=pros_cons,
            verdict=verdict,
            meta={"session_id": session_id, "trace_id": result.get("trace_id", "")},
        )

    except Exception as e:
        logger.error("compare_error", error=str(e))
        raise HTTPException(status_code=500, detail="Comparison failed") from e
