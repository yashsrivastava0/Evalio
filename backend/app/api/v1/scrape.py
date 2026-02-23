"""Scrape API router — triggers async Celery task.

POST /v1/scrape → ScrapeRequest → Celery task → ScrapeResponse
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.scrape import ScrapeRequest, ScrapeResponse
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/scrape", tags=["scrape"])


@router.post("", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest) -> ScrapeResponse:
    """Trigger product scraping.

    1. Check fetch_status in Redis — reject if already fetching
    2. Check freshness — reject if data is fresh enough
    3. Dispatch Celery task
    4. Set fetch_status to 'fetching' in Redis
    5. Return task_id for polling
    """
    logger.info(
        "scrape_request",
        url=request.url,
        slug=request.product_slug,
        vendor=request.vendor,
    )

    if not request.url and not request.product_slug:
        raise HTTPException(
            status_code=400,
            detail="Provide either url or product_slug",
        )

    try:
        # TODO: Wire to Redis fetch_status check + Celery dispatch
        return ScrapeResponse(
            status="accepted",
            task_id=None,
            product_slug=request.product_slug,
            message="Scrape engine ready — awaiting Celery integration",
            eta_seconds=30,
        )

    except Exception as e:
        logger.error("scrape_error", error=str(e))
        raise HTTPException(status_code=500, detail="Scrape trigger failed") from e
