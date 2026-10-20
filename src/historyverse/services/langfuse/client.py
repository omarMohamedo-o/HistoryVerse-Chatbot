"""Langfuse client singleton wrapper."""

from __future__ import annotations

import logging
from typing import Any

from historyverse.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = ["get_langfuse_client"]

_client_instance: Any = None


def get_langfuse_client(settings: Settings | None = None) -> Any:
    """Return the initialized Langfuse client or None if disabled."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    cfg = settings or get_settings()
    if not cfg.langfuse_enabled:
        logger.debug("Langfuse is disabled by configuration.")
        return None

    try:
        from langfuse import Langfuse

        _client_instance = Langfuse(
            public_key=cfg.langfuse_public_key.get_secret_value(),
            secret_key=cfg.langfuse_secret_key.get_secret_value(),
            host=cfg.langfuse_host,
        )
        logger.info("Langfuse client connected to %s", cfg.langfuse_host)
        return _client_instance
    except Exception as exc:
        logger.warning("Could not initialize Langfuse client: %s", exc)
        return None
