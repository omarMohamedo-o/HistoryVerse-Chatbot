"""Alerting and incident notification dispatcher."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from historyverse.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["AlertNotifier"]


class AlertNotifier:
    """Dispatches operational alerts to webhooks, logs, and monitoring channels."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._webhook_url = self._settings.alert_webhook_url.get_secret_value()
        self._last_alert_time: dict[str, float] = {}
        self._rate_limit = self._settings.alert_rate_limit_seconds

    async def emit_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Dispatch an alert if not rate-limited."""
        now = time.time()
        last_sent = self._last_alert_time.get(alert_type, 0.0)

        if now - last_sent < self._rate_limit:
            logger.debug("Alert '%s' suppressed due to rate limiting.", alert_type)
            return False

        self._last_alert_time[alert_type] = now
        logger.warning(
            "ALERT [%s] severity=%s: %s (metadata=%s)",
            alert_type,
            severity,
            message,
            metadata,
        )

        if not self._webhook_url:
            return True

        payload = {
            "service": self._settings.service_name,
            "environment": self._settings.environment,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self._webhook_url, json=payload)
                if resp.status_code < 400:
                    logger.info("Alert dispatched to webhook: %s", alert_type)
                    return True
                logger.error("Alert webhook rejected with status: %d", resp.status_code)
                return False
        except Exception as exc:
            logger.error("Failed to dispatch alert webhook: %s", exc)
            return False
