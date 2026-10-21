#!/usr/bin/env bash
set -euo pipefail

PRUNE_UNTIL_HOURS="${PRUNE_UNTIL_HOURS:-24}"
DRY_RUN="${DRY_RUN:-false}"

echo "Starting Docker maintenance and cleanup..."
echo "Pruning resources older than ${PRUNE_UNTIL_HOURS} hours (dry-run: ${DRY_RUN})"

docker system df || true

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry-run mode enabled; skipping pruning operations."
  exit 0
fi

docker container prune -f --filter "until=${PRUNE_UNTIL_HOURS}h" || true
docker image prune -af --filter "until=${PRUNE_UNTIL_HOURS}h" || true
docker network prune -f || true
docker builder prune -af --filter "until=${PRUNE_UNTIL_HOURS}h" || true

echo "Docker maintenance complete. Current storage utilization:"
docker system df || true
