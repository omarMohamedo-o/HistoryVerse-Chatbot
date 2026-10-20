"""Experiment tracking and telemetry tracing utilities."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from historyverse.config.settings import get_settings
from historyverse.services.langfuse.client import get_langfuse_client

logger = logging.getLogger(__name__)

__all__ = ["trace_execution", "traced"]


def traced(name: str | None = None) -> Callable:
    """Decorator to trace function execution in Langfuse and structured logs."""

    def decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            client = get_langfuse_client()
            trace = None
            if client:
                try:
                    trace = client.trace(name=op_name)
                except Exception as exc:
                    logger.debug("Langfuse trace creation failed: %s", exc)

            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info(
                    "Completed %s in %.2fms",
                    op_name,
                    duration * 1000,
                    extra={"operation": op_name, "duration_ms": duration * 1000},
                )
                return result
            except Exception as exc:
                duration = time.perf_counter() - start
                logger.error(
                    "Failed %s after %.2fms: %s",
                    op_name,
                    duration * 1000,
                    exc,
                    exc_info=True,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                logger.info("Completed %s in %.2fms", op_name, duration * 1000)
                return result
            except Exception as exc:
                duration = time.perf_counter() - start
                logger.error("Failed %s: %s", op_name, exc, exc_info=True)
                raise

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_execution(
    name: str,
    input_data: Any,
    output_data: Any,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Record a synchronous or asynchronous execution event to Langfuse."""
    client = get_langfuse_client()
    if not client:
        return None

    try:
        trace = client.trace(
            name=name,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
        )
        return getattr(trace, "id", None)
    except Exception as exc:
        logger.debug("Langfuse logging error: %s", exc)
        return None
