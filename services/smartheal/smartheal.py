"""Automated container health restarter with dependency awareness and quarantine."""

import json
import logging
import os
import subprocess
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smartheal")

PROJECT = os.getenv("COMPOSE_PROJECT_NAME", "historyverse")
STATE_PATH = os.getenv("STATE_PATH", "/state/smartheal_state.json")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
BASE_BACKOFF_SECONDS = int(os.getenv("BASE_BACKOFF_SECONDS", "10"))
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
MAX_RESTARTS_IN_WINDOW = int(os.getenv("MAX_RESTARTS_IN_WINDOW", "5"))
WINDOW_SECONDS = int(os.getenv("WINDOW_SECONDS", "600"))

# Service restart dependency graph
DEPENDENCIES: dict[str, list[str]] = {
    "historyverse-api": ["redis"],
    "historyverse-caddy": ["historyverse-api"],
}


def load_state() -> dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    temp_path = f"{STATE_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(temp_path, STATE_PATH)


def get_unhealthy_containers() -> list[str]:
    try:
        cmd = [
            "docker",
            "ps",
            "--filter",
            "health=unhealthy",
            "--format",
            "{{.Names}}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    except Exception as exc:
        logger.warning("Error querying Docker health: %s", exc)
        return []


def restart_container(container_name: str) -> bool:
    logger.info("Triggering controlled restart for unhealthy container: %s", container_name)
    try:
        subprocess.run(["docker", "restart", container_name], check=True, timeout=30)
        logger.info("Successfully restarted %s", container_name)
        return True
    except Exception as exc:
        logger.error("Failed to restart %s: %s", container_name, exc)
        return False


def main() -> None:
    logger.info("Smartheal daemon started for project=%s", PROJECT)
    state = load_state()

    while True:
        try:
            now = time.time()
            unhealthy = get_unhealthy_containers()

            for container in unhealthy:
                c_state = state.setdefault(container, {"restarts": [], "quarantined_until": 0})
                if now < c_state["quarantined_until"]:
                    logger.warning(
                        "Container %s is in quarantine until %s; skipping restart.",
                        container,
                        c_state["quarantined_until"],
                    )
                    continue

                recent_restarts = [t for t in c_state["restarts"] if now - t < WINDOW_SECONDS]
                if len(recent_restarts) >= MAX_RESTARTS_IN_WINDOW:
                    quarantine_until = now + 1800
                    c_state["quarantined_until"] = quarantine_until
                    logger.error(
                        "Container %s exceeded max restarts (%d). Quarantining for 30m.",
                        container,
                        len(recent_restarts),
                    )
                    save_state(state)
                    continue

                if restart_container(container):
                    recent_restarts.append(now)
                    c_state["restarts"] = recent_restarts
                    save_state(state)

        except Exception as exc:
            logger.error("Unexpected error in smartheal loop: %s", exc)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
