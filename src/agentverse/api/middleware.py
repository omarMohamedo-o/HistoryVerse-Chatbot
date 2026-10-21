"""HTTP middleware for structured request logging and error handling."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from agentverse.core.exceptions import HistoryVerseError

logger = logging.getLogger(__name__)

__all__ = ["ErrorHandlingMiddleware", "RequestLoggingMiddleware"]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests, response status, and duration."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        logger.info(
            "HTTP request start: [%s] %s %s",
            req_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start_time) * 1000
            response.headers["X-Request-ID"] = req_id
            logger.info(
                "HTTP request complete: [%s] status=%d duration=%.2fms",
                req_id,
                response.status_code,
                elapsed,
            )
            return response
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(
                "HTTP request error: [%s] duration=%.2fms error=%s",
                req_id,
                elapsed,
                exc,
                exc_info=True,
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Converts uncaught domain exceptions into standard JSON error responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except HistoryVerseError as exc:
            logger.warning("Domain exception handled: %s (%s)", type(exc).__name__, exc)
            return JSONResponse(
                status_code=400,
                content={"error": type(exc).__name__, "message": str(exc)},
            )
        except Exception as exc:
            logger.error("Unhandled internal server error: %s", exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred.",
                },
            )
