"""Security, tenant isolation, credentials management, and API authorization test suite."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from agent.api.dependencies import get_app_settings, verify_api_key
from agent.api.routes.agents import CreateAgentRequest
from agent.config.settings import Settings
from agent.services.agents.registry_store import AgentRegistryStore
from agent.services.agents.team_store import TeamRegistryStore
from agent.services.memory.agent_memory import AgentMemoryManager


def test_api_key_authentication_matrix():
    app = FastAPI()

    @app.get("/secure-endpoint", dependencies=[Depends(verify_api_key)])
    def secure_endpoint():
        return {"status": "authenticated"}

    # 1. Auth Disabled -> Passes without credentials
    settings_no_auth = Settings(auth_enabled=False)
    app.dependency_overrides[get_app_settings] = lambda: settings_no_auth
    client = TestClient(app)
    resp = client.get("/secure-endpoint")
    assert resp.status_code == 200
    assert resp.json() == {"status": "authenticated"}

    # 2. Auth Enabled -> Missing credentials raises 401
    settings_auth = Settings(auth_enabled=True, api_key=SecretStr("super-secret-key-1234"))
    app.dependency_overrides[get_app_settings] = lambda: settings_auth

    resp_missing = client.get("/secure-endpoint")
    assert resp_missing.status_code == 401
    assert "Missing API Key" in resp_missing.json()["detail"]

    # 3. Invalid X-API-Key -> 403 Forbidden
    resp_invalid = client.get("/secure-endpoint", headers={"X-API-Key": "wrong-key"})
    assert resp_invalid.status_code == 403
    assert "Invalid API Key" in resp_invalid.json()["detail"]

    # 4. Valid X-API-Key -> 200 OK
    resp_valid = client.get("/secure-endpoint", headers={"X-API-Key": "super-secret-key-1234"})
    assert resp_valid.status_code == 200
    assert resp_valid.json() == {"status": "authenticated"}

    # 5. Invalid Bearer token -> 403 Forbidden
    resp_invalid_bearer = client.get(
        "/secure-endpoint", headers={"Authorization": "Bearer wrong-bearer-key"}
    )
    assert resp_invalid_bearer.status_code == 403

    # 6. Valid Bearer token -> 200 OK
    resp_valid_bearer = client.get(
        "/secure-endpoint", headers={"Authorization": "Bearer super-secret-key-1234"}
    )
    assert resp_valid_bearer.status_code == 200

    # 7. Malformed Authorization header without bearer prefix -> 401
    resp_malformed = client.get("/secure-endpoint", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert resp_malformed.status_code == 401


def test_dependencies_cache_and_singletons():
    from unittest.mock import MagicMock, patch

    from agent.api.dependencies import (
        get_agent_registry_store,
        get_drift_detector,
        reset_dependencies_cache,
    )

    reset_dependencies_cache()

    # 1. get_agent_registry_store singleton instantiation
    store = get_agent_registry_store(Settings())
    assert store is not None
    assert get_agent_registry_store(Settings()) is store

    # 2. get_drift_detector with fallback failure
    mock_prov = MagicMock()
    settings_fallback = Settings(drift_fallback_model="nonexistent-model")
    with patch(
        "agent.services.llm.factory.LLMFactory.create", side_effect=Exception("Model load fail")
    ):
        detector = get_drift_detector(settings=settings_fallback, provider=mock_prov)
        assert detector is not None

    reset_dependencies_cache()


def test_credentials_sanitization_and_no_leakage(tmp_path):
    """Ensure raw LLM provider API keys are masked in metadata and never leaked."""
    db_file = tmp_path / "agents.db"
    store = AgentRegistryStore(db_path=str(db_file))

    raw_secret_key = "sk-live-super-secret-openai-token-998877"
    created = store.save_agent(
        agent_id="classified-agent",
        name="Security Analyst",
        role="Auditor",
        system_prompt="Analyze code safely.",
        domain="security",
        provider_name="openai",
        model_name="gpt-4o",
        temperature=0.2,
        metadata={"api_key": raw_secret_key, "token": "secret_abc", "classification": "restricted"},
    )

    # In stored record, sensitive metadata keys must be automatically masked as '******'
    assert created["metadata"]["api_key"] == "******"
    assert created["metadata"]["token"] == "******"
    assert created["metadata"]["classification"] == "restricted"

    fetched = store.get_agent("classified-agent")
    assert fetched is not None
    assert fetched["agent_id"] == "classified-agent"
    assert fetched["metadata"]["api_key"] == "******"

    # API Request model validation redacts raw strings if formatted as SecretStr
    req = CreateAgentRequest(
        agent_id="classified-agent",
        name="Security Analyst",
        role="Auditor",
        system_prompt="Analyze safely and adhere to security guardrails.",
        provider="openai",
        model_name="gpt-4o",
        api_key=raw_secret_key,
    )
    dumped = req.model_dump()
    assert dumped["agent_id"] == "classified-agent"


@pytest.mark.asyncio
async def test_tenant_and_agent_memory_isolation():
    """Verify strict boundaries between distinct agent memories in multi-agent environments."""
    settings = Settings()
    manager = AgentMemoryManager(settings=settings)

    # Agent A writes secrets and turns
    mem_a = manager.get_agent_memory("agent_alpha")
    await mem_a.add_exchange_async(
        "session_100", "What is the secret vault code?", "The code is 998811."
    )

    # Agent B writes different turns
    mem_b = manager.get_agent_memory("agent_beta")
    await mem_b.add_exchange_async(
        "session_100", "What is the public library schedule?", "Open 9am to 5pm."
    )

    # Agent B query on same session_id MUST NOT receive Agent A's memory
    turns_b = await mem_b.get_history_async("session_100")
    assert len(turns_b) == 2
    assert "library" in turns_b[0]["content"]
    assert "998811" not in str(turns_b)

    turns_a = await mem_a.get_history_async("session_100")
    assert len(turns_a) == 2
    assert "998811" in turns_a[1]["content"]


def test_team_execution_tenant_isolation(tmp_path):
    """Verify team store executions are bound to their respective team definitions and survive lookups."""
    db_file = tmp_path / "teams_security.db"
    store = TeamRegistryStore(db_path=str(db_file))

    store.save_team(
        team_id="finance_team",
        name="Finance Analysts",
        description="Processes invoices",
        strategy="sequential",
        members=["agent_auditor", "agent_cfo"],
    )

    store.save_team(
        team_id="health_team",
        name="Medical Advisors",
        description="Processes diagnostics",
        strategy="parallel",
        members=["agent_doctor"],
    )

    run_fin = store.create_execution(
        run_id="run_fin_001",
        team_id="finance_team",
        strategy="sequential",
        query="Quarterly revenue breakdown",
    )
    run_med = store.create_execution(
        run_id="run_med_001",
        team_id="health_team",
        strategy="parallel",
        query="Blood pressure guidance",
    )

    assert run_fin["team_id"] == "finance_team"
    assert run_med["team_id"] == "health_team"

    # Listing team executions strictly preserves boundary
    runs_fin = store.list_executions(team_id="finance_team")
    assert len(runs_fin) == 1
    assert runs_fin[0]["query"] == "Quarterly revenue breakdown"

    runs_med = store.list_executions(team_id="health_team")
    assert len(runs_med) == 1
    assert runs_med[0]["query"] == "Blood pressure guidance"


def test_cross_tenant_negative_security_checks(tmp_path):
    """Negative security test: Verify tenant cannot read, list, or delete another tenant's agents or teams."""
    agent_store = AgentRegistryStore(db_path=str(tmp_path / "agents_sec.db"))
    team_store = TeamRegistryStore(db_path=str(tmp_path / "teams_sec.db"))

    # 1. Tenant A creates custom agent
    agent_store.save_agent(
        agent_id="tenant_a_agent",
        name="Alpha Auditor",
        role="Auditor",
        system_prompt="Tenant A private logic.",
        metadata={"tenant_id": "tenant_alpha", "confidential": "alpha_data"},
    )

    # Tenant B tries to fetch Tenant A's agent
    assert agent_store.get_agent("tenant_a_agent", tenant_id="tenant_beta") is None
    # Tenant A can fetch its own agent
    assert agent_store.get_agent("tenant_a_agent", tenant_id="tenant_alpha") is not None

    # Tenant B lists agents -> must NOT see Tenant A's agent
    agents_b = agent_store.list_agents(tenant_id="tenant_beta")
    assert len(agents_b) == 0

    # Tenant B attempts malicious deletion of Tenant A's agent -> Refused
    delete_breach = agent_store.delete_agent("tenant_a_agent", tenant_id="tenant_beta")
    assert delete_breach is False
    assert agent_store.get_agent("tenant_a_agent") is not None

    # 2. Tenant A creates team
    team_store.save_team(
        team_id="tenant_a_team",
        name="Alpha Team",
        strategy="sequential",
        members=["tenant_a_agent"],
        config={"tenant_id": "tenant_alpha"},
    )

    # Tenant B tries to fetch Tenant A's team
    assert team_store.get_team("tenant_a_team", tenant_id="tenant_beta") is None
    # Tenant A can fetch its own team
    assert team_store.get_team("tenant_a_team", tenant_id="tenant_alpha") is not None

    # Tenant B lists teams -> must NOT see Tenant A's team
    teams_b = team_store.list_teams(tenant_id="tenant_beta")
    assert len(teams_b) == 0

    # Tenant B attempts malicious deletion of Tenant A's team -> Refused
    team_breach = team_store.delete_team("tenant_a_team", tenant_id="tenant_beta")
    assert team_breach is False
    assert team_store.get_team("tenant_a_team") is not None

    # Legitimate deletion by Tenant A succeeds
    assert agent_store.delete_agent("tenant_a_agent", tenant_id="tenant_alpha") is True
    assert team_store.delete_team("tenant_a_team", tenant_id="tenant_alpha") is True


def test_nested_metadata_and_secret_variants_sanitization(tmp_path):
    """Verify deep recursive metadata sanitization across lists, nested dicts, and casing variations."""
    store = AgentRegistryStore(db_path=str(tmp_path / "agents_nest.db"))

    complex_metadata = {
        "regular_info": "public project",
        "UPPERCASE_KEY": "public value",
        "API_KEY": "sk-top-secret-9988",
        "AUTH_BEARER": "bearer token-123",
        "nested_layer": {
            "client_secret": "xyz_secret",
            "private_key": "-----BEGIN PRIVATE KEY-----",
            "safe_property": 42,
            "deep_nested": {
                "PassWord": "SuperStrongPassword!",
            },
        },
        "credentials_list": ["secret1", "secret2"],
        "service_endpoints": [
            {"API_KEY": "sk-nested-in-list"},
            {"public_note": "ok"},
            "sk-raw-token-in-list",
            "bearer raw-bearer-in-list",
            "safe string in list",
        ],
    }

    saved = store.save_agent(
        agent_id="deep-sanitized-agent",
        name="Deep Sanitized",
        role="Security Test",
        system_prompt="Check nested credentials sanitization.",
        metadata=complex_metadata,
    )

    meta = saved["metadata"]
    assert meta["regular_info"] == "public project"
    assert meta["UPPERCASE_KEY"] == "public value"
    assert meta["API_KEY"] == "******"
    assert meta["AUTH_BEARER"] == "******"

    # Entire list masked when key contains 'credential'
    assert meta["credentials_list"] == "******"

    # Nested dict masking
    assert meta["nested_layer"]["client_secret"] == "******"
    assert meta["nested_layer"]["private_key"] == "******"
    assert meta["nested_layer"]["safe_property"] == 42
    assert meta["nested_layer"]["deep_nested"]["PassWord"] == "******"

    # List of dicts and secret strings masking within neutral list key
    assert meta["service_endpoints"][0]["API_KEY"] == "******"
    assert meta["service_endpoints"][1]["public_note"] == "ok"
    assert meta["service_endpoints"][2] == "******"
    assert meta["service_endpoints"][3] == "******"
    assert meta["service_endpoints"][4] == "safe string in list"

