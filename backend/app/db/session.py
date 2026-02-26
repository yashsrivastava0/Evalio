"""app.db.session — Async DB session context manager for Celery workers.

This module is a thin wrapper around the canonical session factory defined in
app.models.database. It does NOT create a second engine or connection pool.

Usage (Celery tasks):
    from app.db.session import get_async_session

    async with get_async_session() as db:
        result = await db.execute(...)

Architectural contract:
    - One engine, one pool — defined in app.models.database.
    - This module only provides a convenience context manager.
    - FastAPI endpoints should use app.models.database.get_db (FastAPI Depends).
    - Celery workers use this module directly (no FastAPI DI available).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import async_session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session.

    Wraps the canonical async_session_factory from app.models.database.
    Commits on clean exit, rolls back on exception, always closes.

    This is the correct entry point for Celery tasks and any non-FastAPI
    consumers that need a DB session.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
