#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT="${1:-production}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_FILE="$ROOT_DIR/configs/vault/policies/agentverse-read.hcl"
POLICY_NAME="agentverse-deploy-${ENVIRONMENT}"
ROLE_NAME="agentverse-deploy-${ENVIRONMENT}"
MOUNT="${VAULT_KV_MOUNT:-agentverse}"

command -v vault >/dev/null || { echo "vault CLI not found in PATH" >&2; exit 1; }
: "${VAULT_ADDR:?Set VAULT_ADDR (e.g. http://127.0.0.1:8200)}"

echo "[vault-bootstrap] Ensuring KV v2 secrets engine at '${MOUNT}'..."
if ! vault secrets list -format=json | jq -e ".\"${MOUNT}/\"" >/dev/null 2>&1; then
  vault secrets enable -path="${MOUNT}" kv-v2
fi

echo "[vault-bootstrap] Writing policy '${POLICY_NAME}'..."
vault policy write "${POLICY_NAME}" "${POLICY_FILE}"

echo "[vault-bootstrap] Enabling and configuring AppRole '${ROLE_NAME}'..."
vault auth enable approle 2>/dev/null || true
vault write "auth/approle/role/${ROLE_NAME}" \
  token_policies="${POLICY_NAME}" \
  token_ttl=1h \
  token_max_ttl=4h

ROLE_ID="$(vault read -format=json "auth/approle/role/${ROLE_NAME}/role-id" | jq -r '.data.role_id')"
SECRET_ID="$(vault write -f -format=json "auth/approle/role/${ROLE_NAME}/secret-id" | jq -r '.data.secret_id')"

echo "[vault-bootstrap] AppRole initialized successfully."
echo "ROLE_ID=${ROLE_ID}"
echo "SECRET_ID=${SECRET_ID}"
