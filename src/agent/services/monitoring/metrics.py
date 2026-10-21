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
    "VOICE_BARGE_IN_EVENTS",
    "VOICE_SESSIONS_ACTIVE",
    "VOICE_STT_DURATION",
    "VOICE_TTFA",
]

try:
    from prometheus_client import Counter, Gauge, Histogram

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
    VOICE_SESSIONS_ACTIVE = Gauge(
        "agent_voice_sessions_active",
        "Number of currently active real-time voice sessions",
    )
    VOICE_TTFA = Histogram(
        "agent_voice_ttfa_seconds",
        "Time to first audio chunk generated in seconds",
        buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    VOICE_BARGE_IN_EVENTS = Counter(
        "agent_voice_barge_in_total",
        "Total user speech barge-in interruptions detected",
    )
    VOICE_STT_DURATION = Histogram(
        "agent_voice_stt_duration_seconds",
        "STT processing latency in seconds",
    )
except ImportError:
    logger.debug("prometheus_client not available; using mock metric stubs.")

    class _DummyMetric:
        def labels(self, *args, **kwargs) -> Any:
            return self

        def inc(self, *args, **kwargs) -> None:
            pass

        def dec(self, *args, **kwargs) -> None:
            pass

        def set(self, *args, **kwargs) -> None:
            pass

        def observe(self, *args, **kwargs) -> None:
            pass

    REQUEST_COUNT = _DummyMetric()
    REQUEST_LATENCY = _DummyMetric()
    DRIFT_EVENTS = _DummyMetric()
    HALLUCINATION_SCORES = _DummyMetric()
    LLM_LATENCY = _DummyMetric()
    VOICE_SESSIONS_ACTIVE = _DummyMetric()
    VOICE_TTFA = _DummyMetric()
    VOICE_BARGE_IN_EVENTS = _DummyMetric()
    VOICE_STT_DURATION = _DummyMetric()
