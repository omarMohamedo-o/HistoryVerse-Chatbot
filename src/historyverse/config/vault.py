"""HashiCorp Vault secrets resolver for production deployments."""

from __future__ import annotations

import logging
from typing import Any
import httpx

from historyverse.config.settings import Settings

logger = logging.getLogger(__name__)

__all__ = ["VaultSecretsResolver"]


class VaultSecretsResolver:
    """Fetches application credentials from HashiCorp Vault KV v2 mount."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._vault_addr = settings.vault_addr.rstrip("/")
        self._token = settings.vault_token.get_secret_value()
        self._mount = settings.vault_mount
        self._secret_path = settings.vault_path

    async def get_client_token(self) -> str:
        """Resolve token via direct token or AppRole authentication."""
        if self._token:
            return self._token

        role_id = self._settings.vault_role_id
        secret_id = self._settings.vault_secret_id.get_secret_value()

        if role_id and secret_id:
            url = f"{self._vault_addr}/v1/auth/approle/login"
            payload = {"role_id": role_id, "secret_id": secret_id}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                token = resp.json().get("auth", {}).get("client_token", "")
                logger.info("Authenticated with Vault via AppRole.")
                return token

        return ""

    async def fetch_secrets(self) -> dict[str, Any]:
        """Read key-value secrets from Vault KV v2."""
        if not self._settings.vault_enabled:
            return {}

        token = await self.get_client_token()
        if not token:
            logger.warning("Vault is enabled but no token or AppRole credentials provided.")
            return {}

        url = f"{self._vault_addr}/v1/{self._mount}/data/{self._secret_path}"
        headers = {"X-Vault-Token": token}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json().get("data", {}).get("data", {})
                logger.info("Successfully fetched %d secrets from Vault at %s", len(data), self._secret_path)
                return data
        except Exception as exc:
            logger.error("Failed to retrieve secrets from Vault: %s", exc)
            return {}
