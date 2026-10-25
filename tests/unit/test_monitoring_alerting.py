"""Unit tests for AlertNotifier and Prometheus monitoring instruments."""

import httpx
import pytest
import respx
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.services.alerting.notifier import AlertNotifier


@pytest.mark.asyncio
async def test_alert_notifier_no_webhook():
    settings = Settings(alert_webhook_url=SecretStr(""))
    notifier = AlertNotifier(settings)

    dispatched = await notifier.emit_alert(
        alert_type="high_drift",
        message="Model drift score exceeded threshold",
        severity="critical",
    )
    assert dispatched is True


@pytest.mark.asyncio
@respx.mock
async def test_alert_notifier_with_webhook():
    webhook_url = "https://hooks.slack.com/services/T00/B00/X00"
    settings = Settings(alert_webhook_url=SecretStr(webhook_url), alert_rate_limit_seconds=10)
    notifier = AlertNotifier(settings)

    respx.post(webhook_url).mock(return_value=httpx.Response(200, json={"ok": True}))

    dispatched = await notifier.emit_alert(
        alert_type="unlearning_incident",
        message="Unauthorized concept probe detected",
        severity="warning",
    )
    assert dispatched is True

    # Immediate second alert of same type should be rate limited
    second_dispatched = await notifier.emit_alert(
        alert_type="unlearning_incident",
        message="Second probe",
        severity="warning",
    )
    assert second_dispatched is False


def test_prometheus_metrics_stubs():
    import importlib

    import agent.services.monitoring.metrics as m

    importlib.reload(m)
    m.REQUEST_COUNT.labels(endpoint="/v1/chat", status="200").inc()
    m.REQUEST_LATENCY.labels(endpoint="/v1/chat").observe(0.125)
    m.DRIFT_EVENTS.labels(action="reingest").inc()
    m.HALLUCINATION_SCORES.observe(0.05)
    m.LLM_LATENCY.labels(model="gemini-1.5-flash").observe(0.45)
    m.VOICE_SESSIONS_ACTIVE.inc()
    m.VOICE_TTFA.observe(0.3)
    m.VOICE_BARGE_IN_EVENTS.inc()
    m.VOICE_STT_DURATION.observe(0.2)


@pytest.mark.asyncio
@respx.mock
async def test_alert_notifier_errors():
    webhook_url = "https://hooks.slack.com/services/T00/B00/X00"
    settings = Settings(alert_webhook_url=SecretStr(webhook_url), alert_rate_limit_seconds=0)
    notifier = AlertNotifier(settings)

    # 500 error from webhook
    respx.post(webhook_url).mock(return_value=httpx.Response(500, text="Internal Error"))
    assert await notifier.emit_alert("high_drift", "Error test", severity="critical") is False

    # Connection failure
    respx.post(webhook_url).mock(side_effect=httpx.ConnectError("Connection refused"))
    assert await notifier.emit_alert("high_drift", "Fail test", severity="critical") is False


def test_metrics_dummy_metric_coverage():
    from agent.services.monitoring.metrics import _DummyMetric

    dummy = _DummyMetric()
    assert dummy.labels(a="1", b="2") is dummy
    dummy.inc(2)
    dummy.dec(1)
    dummy.set(5)
    dummy.observe(0.5)
