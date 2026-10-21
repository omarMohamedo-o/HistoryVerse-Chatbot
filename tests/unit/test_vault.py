"""Unit tests for HashiCorp Vault secrets resolver."""

import httpx
import pytest
import respx
from pydantic import SecretStr

from agentverse.config.settings import Settings
from agentverse.config.vault import VaultSecretsResolver


@pytest.mark.asyncio
async def test_vault_disabled_by_default():
    settings = Settings(vault_enabled=False)
    resolver = VaultSecretsResolver(settings)
    secrets = await resolver.fetch_secrets()
    assert secrets == {}


@pytest.mark.asyncio
@respx.mock
async def test_vault_token_auth_fetch_success():
    settings = Settings(
        vault_enabled=True,
        vault_addr="http://vault.test:8200",
        vault_token=SecretStr("s.test-token-12345"),
        vault_mount="agentverse",
        vault_path="app",
    )
    resolver = VaultSecretsResolver(settings)

    respx.get("http://vault.test:8200/v1/agentverse/data/app").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"data": {"GOOGLE_API_KEY": "AIzaSySecretGoogleKey"}}},
        )
    )

    secrets = await resolver.fetch_secrets()
    assert secrets == {"GOOGLE_API_KEY": "AIzaSySecretGoogleKey"}


@pytest.mark.asyncio
@respx.mock
async def test_vault_approle_auth_fetch_success():
    settings = Settings(
        vault_enabled=True,
        vault_addr="http://vault.test:8200",
        vault_token=SecretStr(""),
        vault_role_id="role-history-1",
        vault_secret_id=SecretStr("secret-history-uuid"),
        vault_mount="secret",
        vault_path="credentials",
    )
    resolver = VaultSecretsResolver(settings)

    # Mock AppRole login
    respx.post("http://vault.test:8200/v1/auth/approle/login").mock(
        return_value=httpx.Response(
            200,
            json={"auth": {"client_token": "s.generated-approle-token"}},
        )
    )

    # Mock secret read
    respx.get("http://vault.test:8200/v1/secret/data/credentials").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"data": {"OPENAI_API_KEY": "sk-test-secret"}}},
        )
    )

    secrets = await resolver.fetch_secrets()
    assert secrets == {"OPENAI_API_KEY": "sk-test-secret"}


@pytest.mark.asyncio
async def test_vault_enabled_without_credentials():
    settings = Settings(
        vault_enabled=True,
        vault_addr="http://vault.test:8200",
        vault_token=SecretStr(""),
        vault_role_id="",
        vault_secret_id=SecretStr(""),
    )
    resolver = VaultSecretsResolver(settings)
    secrets = await resolver.fetch_secrets()
    assert secrets == {}


@pytest.mark.asyncio
@respx.mock
async def test_vault_network_failure_handling():
    settings = Settings(
        vault_enabled=True,
        vault_addr="http://vault.test:8200",
        vault_token=SecretStr("s.test-token"),
        vault_mount="agentverse",
        vault_path="app",
    )
    resolver = VaultSecretsResolver(settings)

    respx.get("http://vault.test:8200/v1/agentverse/data/app").mock(
        return_value=httpx.Response(500, json={"errors": ["internal vault error"]})
    )

    secrets = await resolver.fetch_secrets()
    assert secrets == {}
