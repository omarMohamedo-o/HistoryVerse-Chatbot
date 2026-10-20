"""HistoryVerse CLI runner."""

from __future__ import annotations

import logging
import uvicorn
from historyverse.config.settings import get_settings

logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Launching HistoryVerse on %s:%d", settings.api_host, settings.api_port)

    uvicorn.run(
        "historyverse.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
