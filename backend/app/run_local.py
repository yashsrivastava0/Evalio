"""Local Bootstrap — Initialize all services for CLI execution.

Responsibilities:
    1. Redis client (session cost + Celery broker)
    2. Celery broker availability check (Issue #2)
    3. DB session factory
    4. LLM client (OpenAI)
    5. Embedding provider
    6. Graph compilation (exact signature match — Issue #6)

Returns a context dict used by cli.py. No global state.
"""

from __future__ import annotations

import sys
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


async def create_app() -> dict[str, Any]:
    """Initialize all services for local CLI execution.

    Returns:
        {
            "graph": compiled LangGraph,
            "redis": async Redis client,
            "db_session": SQLAlchemy async session,
            "llm_client": OpenAI client (or None),
            "embedding_provider": embedding provider instance,
            "session_id": str (uuid4),
            "celery_available": bool,
        }
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    # ── 1. Redis (required for session cost + Celery broker) ──
    try:
        redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=5,
        )
        # Verify connection
        await redis_client.ping()
        print("  ✅ Redis connected")
    except Exception as e:
        print(f"  ❌ Redis connection failed: {e}")
        print("     Redis is required for session cost tracking.")
        sys.exit(1)

    # ── 2. Celery (optional, no startup ping) ──
    # Tasks are queued to Redis broker and processed when a worker is running.
    # Start a worker separately if you need background scraping:
    #   python -m celery -A app.workers.tasks worker --loglevel=info --pool=solo
    celery_available = False  # Detected lazily at dispatch time in fetch_trigger node

    # ── 3. DB session factory + connectivity test ──
    try:
        from app.models.database import async_session_factory
        from sqlalchemy import text as sa_text
        async with async_session_factory() as _test_session:
            await _test_session.execute(sa_text("SELECT 1"))
        db_session = async_session_factory   # pass the factory, not a raw session
        print("  ✅ PostgreSQL connected")
    except Exception as e:
        print(f"  ❌ PostgreSQL connection failed: {e}")
        print(f"     URL: {settings.database_url}")
        print(f"     Make sure PostgreSQL is running on localhost:5432")
        sys.exit(1)

    # ── 4. LLM client (OpenAI) ──
    llm_client = None
    try:
        if settings.openai_api_key:
            from openai import AsyncOpenAI
            llm_client = AsyncOpenAI(api_key=settings.openai_api_key)
            print(f"  ✅ LLM client ready (model: {settings.llm_model})")
        else:
            print("  ⚠️  No OPENAI_API_KEY — LLM will use template fallback")
    except Exception as e:
        print(f"  ⚠️  LLM client init failed: {e} — will use template fallback")

    # ── 5. Embedding provider ──
    embedding_provider = None
    try:
        from app.services.embedding import create_embedding_provider
        embedding_provider = create_embedding_provider()
        print(f"  ✅ Embedding provider ready ({settings.embedding_model})")
    except Exception as e:
        print(f"  ⚠️  Embedding provider init failed: {e}")

    # ── 6. Graph compilation (Issue #6: exact signature match) ──
    from app.graph import compile_graph
    graph = compile_graph(
        redis_client=redis_client,
        db_session=db_session,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        celery_available=celery_available,
    )
    print("  ✅ LangGraph compiled")

    session_id = str(uuid4())

    return {
        "graph": graph,
        "redis": redis_client,
        "db_session": db_session,
        "llm_client": llm_client,
        "embedding_provider": embedding_provider,
        "session_id": session_id,
        "celery_available": celery_available,
    }


async def shutdown(ctx: dict[str, Any]) -> None:
    """Graceful shutdown — close connections (Adjustment #2)."""
    try:
        db = ctx.get("db_session")
        if db:
            await db.close()
    except Exception:
        pass

    try:
        redis = ctx.get("redis")
        if redis:
            await redis.close()
    except Exception:
        pass

    print("\n  Connections closed. Goodbye!")
