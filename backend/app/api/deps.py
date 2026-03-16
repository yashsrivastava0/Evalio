"""FastAPI dependency injection functions.

Shared dependencies for all v1 endpoints.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import async_session_factory


async def get_redis(request: Request) -> Any:
    """Return the Redis connection pool from app state."""
    return request.app.state.redis


async def get_llm_client(request: Request) -> Any:
    """Return the shared AsyncOpenAI client from app state."""
    return request.app.state.llm_client


async def get_embedding_provider(request: Request) -> Any:
    """Return the shared embedding provider from app state."""
    return request.app.state.embedding_provider


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session."""
    async with async_session_factory() as session:
        yield session
