"""Unit tests for monitoring metrics module, testing both Prometheus and fallback dummy metrics."""

import importlib
import sys
from unittest.mock import MagicMock

import agent.services.monitoring.metrics as metrics_mod


def test_dummy_metrics_fallback_behavior():
    """Verify dummy metrics stubs when prometheus_client is absent."""
    dummy = metrics_mod._DummyMetric()
    # All methods should safely execute without raising errors
    assert dummy.labels("endpoint", "status") is dummy
    assert dummy.inc(1) is None
    assert dummy.dec(1) is None
    assert dummy.set(5) is None
    assert dummy.observe(0.42) is None


def test_prometheus_metrics_when_installed(monkeypatch):
    """Verify that when prometheus_client is available, real metric instruments are created."""
    mock_prom = MagicMock()
    mock_prom.Counter = MagicMock(side_effect=lambda name, desc, *args, **kwargs: f"Counter({name})")
    mock_prom.Gauge = MagicMock(side_effect=lambda name, desc, *args, **kwargs: f"Gauge({name})")
    mock_prom.Histogram = MagicMock(side_effect=lambda name, desc, *args, **kwargs: f"Histogram({name})")

    monkeypatch.setitem(sys.modules, "prometheus_client", mock_prom)
    reloaded = importlib.reload(metrics_mod)

    try:
        assert reloaded.REQUEST_COUNT == "Counter(agent_requests_total)"
        assert reloaded.REQUEST_LATENCY == "Histogram(agent_request_duration_seconds)"
        assert reloaded.DRIFT_EVENTS == "Counter(agent_drift_events_total)"
        assert reloaded.HALLUCINATION_SCORES == "Histogram(agent_hallucination_score)"
        assert reloaded.LLM_LATENCY == "Histogram(agent_llm_generation_seconds)"
        assert reloaded.VOICE_SESSIONS_ACTIVE == "Gauge(agent_voice_sessions_active)"
        assert reloaded.VOICE_TTFA == "Histogram(agent_voice_ttfa_seconds)"
        assert reloaded.VOICE_BARGE_IN_EVENTS == "Counter(agent_voice_barge_in_total)"
        assert reloaded.VOICE_STT_DURATION == "Histogram(agent_voice_stt_duration_seconds)"
    finally:
        # Reload back to normal environment state
        monkeypatch.delitem(sys.modules, "prometheus_client", raising=False)
        importlib.reload(metrics_mod)
