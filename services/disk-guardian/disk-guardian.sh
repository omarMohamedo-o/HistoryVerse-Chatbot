#!/usr/bin/env bash
set -euo pipefail

HOST_MOUNT="${HOST_MOUNT:-/host}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
WARN_PCT="${DISK_WARN_PCT:-85}"
CRIT_PCT="${DISK_CRIT_PCT:-90}"
INODE_WARN_PCT="${INODE_WARN_PCT:-85}"
INODE_CRIT_PCT="${INODE_CRIT_PCT:-90}"
PRUNE_UNTIL_HOURS="${PRUNE_UNTIL_HOURS:-24}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://alertmanager:9093}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/app/outputs}"

prune_docker() {
  echo "[disk-guardian] Executing safe Docker cleanup..."
  docker system df || true

  docker container prune -f --filter "until=${PRUNE_UNTIL_HOURS}h" || true
  docker image prune -af --filter "until=${PRUNE_UNTIL_HOURS}h" || true
  docker network prune -f || true
  docker builder prune -af --filter "until=${PRUNE_UNTIL_HOURS}h" || true
}

clean_transient_media() {
  if [[ -d "${OUTPUTS_DIR}" ]]; then
    echo "[disk-guardian] Cleaning transient audio and temporary exports older than 24h in ${OUTPUTS_DIR}"
    find "${OUTPUTS_DIR}" -type f \( -name "*.mp3" -o -name "*.wav" -o -name "*.tmp" \) -mtime +1 -delete 2>/dev/null || true
  fi
}

send_alert() {
  local severity="$1"
  local summary="$2"
  local description="$3"

  local now end
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  end="$(date -u -d '+15 minutes' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)"

  curl -fsS -XPOST "${ALERTMANAGER_URL}/api/v2/alerts" \
    -H 'Content-Type: application/json' \
    -d "[{
      \"labels\": {
        \"alertname\": \"DiskGuardCleanupTriggered\",
        \"severity\": \"${severity}\",
        \"job\": \"disk-guardian\",
        \"instance\": \"host\"
      },
      \"annotations\": {
        \"summary\": \"${summary}\",
        \"description\": \"${description}\"
      },
      \"startsAt\": \"${now}\",
      \"endsAt\": \"${end}\"
    }]" >/dev/null 2>&1 || true
}

get_df_pct() {
  df -P "${HOST_MOUNT}" | awk 'NR==2{gsub(/%/,"",$5);print $5}'
}

get_inode_pct() {
  df -Pi "${HOST_MOUNT}" | awk 'NR==2{gsub(/%/,"",$5);print $5}'
}

echo "[disk-guardian] Starting Disk Guard service monitoring ${HOST_MOUNT} every ${CHECK_INTERVAL_SECONDS}s"

while true; do
  if [[ ! -d "${HOST_MOUNT}" ]]; then
    echo "[disk-guardian] Host mount ${HOST_MOUNT} not present; monitoring root filesystem." >&2
    disk_pct="$(df -P / | awk 'NR==2{gsub(/%/,"",$5);print $5}')"
    inode_pct="$(df -Pi / | awk 'NR==2{gsub(/%/,"",$5);print $5}')"
  else
    disk_pct="$(get_df_pct || echo 0)"
    inode_pct="$(get_inode_pct || echo 0)"
  fi

  echo "[disk-guardian] Storage status: disk=${disk_pct}% inode=${inode_pct}%"

  triggered=false
  severity="warning"

  if [[ "${disk_pct}" -ge "${CRIT_PCT}" || "${inode_pct}" -ge "${INODE_CRIT_PCT}" ]]; then
    severity="critical"
    triggered=true
  elif [[ "${disk_pct}" -ge "${WARN_PCT}" || "${inode_pct}" -ge "${INODE_WARN_PCT}" ]]; then
    severity="warning"
    triggered=true
  fi

  if [[ "${triggered}" == "true" ]]; then
    echo "[disk-guardian] Disk usage threshold exceeded. Running automated pruning procedures."
    prune_docker
    clean_transient_media
    send_alert "${severity}" \
      "Disk Guard autonomy cleanup triggered" \
      "Storage reached disk=${disk_pct}% inode=${inode_pct}%. Executed container, image, and cache pruning."
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
