"""Dead-man's switch heartbeat ping to external monitoring service."""

import logging
import os
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("heartbeat")

HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "")
HEARTBEAT_URL_FILE = os.getenv("HEARTBEAT_URL_FILE", "/run/heartbeat/url")
PROMETHEUS_READY_URL = os.getenv("PROMETHEUS_READY_URL", "http://prometheus:9090/-/ready")
POLL_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "60"))


def get_heartbeat_url() -> str:
    if HEARTBEAT_URL:
        return HEARTBEAT_URL.strip()
    if os.path.exists(HEARTBEAT_URL_FILE):
        try:
            with open(HEARTBEAT_URL_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


def check_prometheus_ready() -> bool:
    try:
        req = urllib.request.Request(PROMETHEUS_READY_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.debug("Prometheus readiness probe failed: %s", exc)
        return False


def ping_heartbeat(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "HistoryVerse-Heartbeat/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Heartbeat ping dispatched successfully (HTTP %d)", resp.status)
            return True
    except Exception as exc:
        logger.warning("Heartbeat ping dispatch failed: %s", exc)
        return False


def main() -> None:
    logger.info("Heartbeat daemon initiated.")
    while True:
        url = get_heartbeat_url()
        if not url:
            logger.debug("No heartbeat URL configured. Sleeping.")
        else:
            prom_ok = check_prometheus_ready()
            if prom_ok:
                ping_heartbeat(url)
            else:
                logger.warning("Internal health checks not passing; withholding heartbeat ping to trigger external alert.")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
