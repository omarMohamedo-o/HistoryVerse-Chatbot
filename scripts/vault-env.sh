#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENVIRONMENT="${1:-development}"
shift || true

OUT_FILE="$ROOT_DIR/.env.runtime"
MOUNT="${VAULT_KV_MOUNT:-historyverse}"

while (($#)); do
  case "$1" in
    --out) OUT_FILE="$2"; shift 2 ;;
    --mount) MOUNT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: scripts/vault-env.sh <development|production> [--out FILE] [--mount MOUNT]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v vault >/dev/null || { echo "vault CLI not found in PATH" >&2; exit 1; }
: "${VAULT_ADDR:?Set VAULT_ADDR (e.g. http://127.0.0.1:8200)}"

SECRET_PATH="${MOUNT}/data/${ENVIRONMENT}/app"
if ! vault kv get "${SECRET_PATH}" >/dev/null 2>&1; then
  SECRET_PATH="${MOUNT}/data/app"
fi

echo "[vault-env] Reading secrets from ${SECRET_PATH}..."

mkdir -p "$(dirname "$OUT_FILE")"
touch "$OUT_FILE"
chmod 0600 "$OUT_FILE"

vault kv get -format=json "${SECRET_PATH}" | jq -r '.data.data | to_entries[] | "\(.key)=\(.value)"' > "$OUT_FILE"

echo "[vault-env] Successfully wrote secrets to $OUT_FILE"
