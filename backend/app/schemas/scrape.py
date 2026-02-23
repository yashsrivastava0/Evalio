"""Pydantic schemas for /v1/scrape endpoint.

Triggers async scraper via Celery — never blocks user.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScrapeRequest(BaseModel):
    """Request to scrape a product."""

    url: str | None = Field(None, description="Product URL to scrape")
    product_slug: str | None = Field(None, description="Product slug to refresh")
    vendor: str | None = Field(None, description="Vendor hint (e.g., 'amazon')")
    force_refresh: bool = Field(False, description="Force re-scrape even if data is fresh")


class ScrapeResponse(BaseModel):
    """Scrape response — always async, returns task status."""

    status: str = Field(description="'accepted' | 'already_fresh' | 'already_fetching'")
    task_id: str | None = Field(None, description="Celery task ID for tracking")
    product_slug: str | None = None
    message: str = ""
    eta_seconds: int | None = Field(None, description="Estimated time to completion")
