"""Prometheus metrics instruments for Agent."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DRIFT_EVENTS",
    "HALLUCINATION_SCORES",
    "LLM_LATENCY",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
]

try:
    from prometheus_client import Counter, Histogram

    REQUEST_COUNT = Counter(
        "agent_requests_total",
        "Total processed API requests",
        ["endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "agent_request_duration_seconds",
        "HTTP request duration in seconds",
        ["endpoint"],
    )
    DRIFT_EVENTS = Counter(
        "agent_drift_events_total",
        "Total model drift events triggered",
        ["action"],
    )
    HALLUCINATION_SCORES = Histogram(
        "agent_hallucination_score",
        "Histogram of computed hallucination scores",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    LLM_LATENCY = Histogram(
        "agent_llm_generation_seconds",
        "LLM inference and generation time in seconds",
        ["model"],
    )
except ImportError:
    logger.debug("prometheus_client not available; using mock metric stubs.")

    class _DummyMetric:
        def labels(self, *args, **kwargs) -> Any:
            return self

        def inc(self, *args, **kwargs) -> None:
            pass

        def observe(self, *args, **kwargs) -> None:
            pass

    REQUEST_COUNT = _DummyMetric()
    REQUEST_LATENCY = _DummyMetric()
    DRIFT_EVENTS = _DummyMetric()
    HALLUCINATION_SCORES = _DummyMetric()
    LLM_LATENCY = _DummyMetric()
