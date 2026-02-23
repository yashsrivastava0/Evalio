"""Structured JSON logging with correlation ID middleware.

Every log line emits JSON with a correlation_id for request tracing.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any

import logging
import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# ── Context var for per-request correlation ID ──
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON rendering."""
    # Convert string level to integer (e.g., "INFO" -> 20)
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_correlation_id,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_correlation_id(
    logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject correlation_id into every log event."""
    cid = correlation_id_ctx.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name) if name else structlog.get_logger()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique correlation_id to every HTTP request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        correlation_id_ctx.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response
