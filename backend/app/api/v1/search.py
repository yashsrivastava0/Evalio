"""Search API router — wraps the unified chat graph.

POST /v1/search → SearchRequest → LangGraph → SearchResponse
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_embedding_provider, get_llm_client, get_redis
from app.graph import compile_graph
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchMeta,
    RankedProduct,
    ProductResult,
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    redis: object = Depends(get_redis),
    db: AsyncSession = Depends(get_db),
    llm_client: object = Depends(get_llm_client),
    embedding_provider: object = Depends(get_embedding_provider),
) -> SearchResponse:
    """Search products via the AI Product Decision Engine.

    Delegates to the unified graph. Maps recommendation_cards component
    output into the SearchResponse format.
    """
    logger.info(
        "search_request",
        query=request.query[:100],
        category=request.category,
        budget_max=request.budget_max,
    )

    session_id = request.session_id or str(uuid4())

    try:
        graph = compile_graph(
            redis_client=redis,
            db_session=db,
            llm_client=llm_client,
            embedding_provider=embedding_provider,
        )
        result = await graph.ainvoke({
            "query": request.query,
            "session_id": session_id,
        })

        structured = result.get("structured_output") or {}
        components = structured.get("components", [])

        # Extract recommendation_cards component for ranked results
        ranked_products: list[RankedProduct] = []
        for component in components:
            if component.get("component") == "recommendation_cards":
                for i, p in enumerate(component.get("products", []), start=1):
                    ranked_products.append(RankedProduct(
                        rank=i,
                        product=ProductResult(
                            product_id=p.get("id") or 0,
                            product_slug=p.get("slug") or "",
                            brand="",
                            model_name="",
                            canonical_title=p.get("title") or "",
                            category=request.category.value if request.category else "",
                            price_inr=p.get("price"),
                        ),
                        scores={"match_score": p.get("match_score", 0.0), "breakdown": p.get("match_score_breakdown", {})},  # type: ignore[arg-type]
                        explanation_snippet=p.get("reason") or "",
                    ))
                break

        retrieval_route = result.get("retrieval_route") or "ai"
        cache_hit = retrieval_route == "cache_hit"

        return SearchResponse(
            results=ranked_products,
            meta=SearchMeta(
                total_candidates=len(ranked_products),
                total_filtered=len(ranked_products),
                total_ranked=len(ranked_products),
                page=request.page,
                limit=request.limit,
                cache_hit=cache_hit,
                cache_type="semantic" if cache_hit else None,
                route=retrieval_route,
                timing_ms={},
            ),
            explanation={"rationale": next(
                (c.get("rationale", "") for c in components if c.get("component") == "recommendation_cards"),
                "",
            )},
        )

    except Exception as e:
        logger.error("search_error", error=str(e))
        raise HTTPException(status_code=500, detail="Search failed") from e
