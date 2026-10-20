# HashiCorp Vault Integration

This document outlines the secret management architecture in HistoryVerse.

## Overview

In alignment with production security governance and zero-trust principles:
- **No secrets in git diffs or committed files**: `.env` is reserved strictly for local ephemeral developer environments.
- **Production secrets are served from Vault**: Sensitive keys (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `LANGFUSE_SECRET_KEY`) reside in a HashiCorp Vault KV v2 mount (`historyverse/`).
- **Dynamic or File-based Ingestion**:
  1. **Runtime injection via `scripts/vault-env.sh`**: Directly pulls secrets into a restricted `0600` `.env.runtime` file upon host deployment.
  2. **In-application client (`VaultSecretsResolver`)**: Directly queries the Vault REST API via AppRole authentication on service startup when `VAULT_ENABLED=true`.

## Bootstrap Workflow

Run once per environment to establish policies and AppRole credentials:

```bash
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="<admin-token>"
./scripts/vault-bootstrap.sh production
```

This generates `ROLE_ID` and `SECRET_ID` to be configured in container environments or GitHub Actions repository secrets.

## Rendering Secrets

```bash
./scripts/vault-env.sh production --out .env.runtime
```
