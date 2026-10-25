"""Integration and unit tests for deep multi-tenant boundary enforcement and credential binding."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.dependencies import verify_api_key
from agent.config.settings import Settings
from agent.services.agents.orchestrator import AgentOrchestrator
from agent.services.agents.registry_store import AgentRegistryStore
from agent.services.agents.team_store import TeamRegistryStore


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.mark.asyncio
async def test_verify_api_key_tenant_binding():
    """Verify tenant credential binding and anti-spoofing in verify_api_key."""
    settings = Settings(auth_enabled=True, api_key="secret_token")

    # 1. Matching tenant binding
    authorized = await verify_api_key(
        x_api_key="tenant_alpha:secret_token",
        authorization=None,
        x_tenant_id="tenant_alpha",
        settings=settings,
    )
    assert authorized is True

    # 2. Key with no tenant header provided (allowed, falls back to raw key)
    authorized_no_header = await verify_api_key(
        x_api_key="tenant_alpha:secret_token",
        authorization=None,
        x_tenant_id=None,
        settings=settings,
    )
    assert authorized_no_header is True

    # 3. Spoofed tenant header - should raise 403 Forbidden
    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(
            x_api_key="tenant_alpha:secret_token",
            authorization=None,
            x_tenant_id="tenant_beta",
            settings=settings,
        )
    assert exc_info.value.status_code == 403
    assert "Tenant mismatch" in exc_info.value.detail

    # 4. Bearer token format with tenant binding
    authorized_bearer = await verify_api_key(
        x_api_key=None,
        authorization="Bearer tenant_alpha:secret_token",
        x_tenant_id="tenant_alpha",
        settings=settings,
    )
    assert authorized_bearer is True

    # 5. Invalid secret key with tenant prefix
    with pytest.raises(HTTPException) as exc_invalid:
        await verify_api_key(
            x_api_key="tenant_alpha:wrong_secret",
            authorization=None,
            x_tenant_id="tenant_alpha",
            settings=settings,
        )
    assert exc_invalid.value.status_code == 403
    assert exc_invalid.value.detail == "Invalid API Key."


@pytest.mark.asyncio
async def test_team_store_and_orchestrator_multi_tenant_boundaries(tmp_path):
    """Test team_store and orchestrator tenant boundary checks directly."""
    db_path = str(tmp_path / "test_tenant.db")
    team_store = TeamRegistryStore(db_path=db_path)
    agent_store = AgentRegistryStore(db_path=db_path)

    settings = Settings(llm_provider="local", agent_persona_id="historical_agent")
    orchestrator = AgentOrchestrator(
        settings=settings,
        store=agent_store,
        team_store=team_store,
    )

    # Register dummy builtin agent to test builtin pass-through
    dummy_builtin = MagicMock()
    dummy_builtin.agent_id = "historical_agent"
    orchestrator.register_agent(dummy_builtin)

    # 1. Register custom agent under tenant_A
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    agent = orchestrator.create_custom_agent(
        agent_id="scholar_a",
        name="Tenant A Scholar",
        role="Researcher",
        domain="history",
        system_prompt="You are a scholar.",
        llm_provider=mock_llm,
        metadata={"tenant_id": "tenant_A"},
    )
    agent_id = agent.agent_id

    # 2. Verify get_agent tenant isolation
    assert orchestrator.get_agent(agent_id, tenant_id="tenant_B") is None
    assert orchestrator.get_agent(agent_id, tenant_id="tenant_A") is not None
    # Builtin agents are accessible by any tenant
    assert orchestrator.get_agent("historical_agent", tenant_id="tenant_B") is not None

    # 3. Verify delete_custom_agent tenant isolation
    # Cross-tenant delete should fail
    deleted_cross = orchestrator.delete_custom_agent(agent_id, tenant_id="tenant_B")
    assert deleted_cross is False
    assert orchestrator.get_agent(agent_id, tenant_id="tenant_A") is not None

    # Authorized tenant delete should succeed
    deleted_auth = orchestrator.delete_custom_agent(agent_id, tenant_id="tenant_A")
    assert deleted_auth is True
    assert orchestrator.get_agent(agent_id, tenant_id="tenant_A") is None

    # 4. Create team under tenant_A
    team_store.save_team(
        team_id="team_alpha",
        name="Alpha Team",
        strategy="sequential",
        members=["historical_agent"],
        config={"tenant_id": "tenant_A"},
    )

    # Record an execution run
    team_store.create_execution(
        run_id="run_alpha_001",
        team_id="team_alpha",
        strategy="sequential",
        query="Research ancient Alexandria",
        status="COMPLETED",
    )

    # 5. Verify get_execution cross-tenant blocking
    exec_cross = team_store.get_execution("run_alpha_001", tenant_id="tenant_B")
    assert exec_cross is None

    exec_auth = team_store.get_execution("run_alpha_001", tenant_id="tenant_A")
    assert exec_auth is not None
    assert exec_auth["run_id"] == "run_alpha_001"

    # 6. Verify list_executions tenant filtering
    execs_tenant_b = team_store.list_executions(tenant_id="tenant_B")
    assert len(execs_tenant_b) == 0

    execs_tenant_a = team_store.list_executions(tenant_id="tenant_A")
    assert len(execs_tenant_a) == 1
    assert execs_tenant_a[0]["run_id"] == "run_alpha_001"

    # 7. Verify run_team_workflow fails for non-existent team
    with pytest.raises(ValueError) as val_err:
        await orchestrator.run_team_workflow("non_existent_team", "query")
    assert "not found" in str(val_err.value)

    # 8. Verify run_team_workflow fails when accessed by unauthorized tenant
    with pytest.raises(ValueError) as val_err_tenant:
        await orchestrator.run_team_workflow("team_alpha", "query", tenant_id="tenant_B")
    assert "not found" in str(val_err_tenant.value)

    # 9. Verify run_individual fails when accessed by unauthorized tenant
    with pytest.raises(ValueError) as val_err_indiv:
        await orchestrator.run_individual("scholar_a", "query", tenant_id="tenant_B")
    assert "not registered" in str(val_err_indiv.value)

    # 10. Execution without team or with non-existent run_id
    assert team_store.get_execution("non_existent_run_id") is None
    team_store.create_execution(
        run_id="run_standalone",
        team_id="",
        strategy="direct",
        query="Standalone run",
        status="COMPLETED",
    )
    assert team_store.get_execution("run_standalone", tenant_id="tenant_B") is not None


def test_api_multi_tenant_routes_enforcement(client):
    """Test full HTTP API endpoint isolation across tenants."""
    headers_a = {"X-API-Key": "dev-local-key", "X-Tenant-ID": "tenant_A"}
    headers_b = {"X-API-Key": "dev-local-key", "X-Tenant-ID": "tenant_B"}

    # 1. Create a custom agent under tenant_A
    create_agent_resp = client.post(
        "/v1/agents/custom",
        headers=headers_a,
        json={
            "agent_id": "scribe-thebes",
            "name": "Scribe of Thebes",
            "role": "Scribe",
            "domain": "egyptology",
            "system_prompt": "You are an Egyptian scribe.",
        },
    )
    assert create_agent_resp.status_code == 200
    agent_id = create_agent_resp.json()["agent_id"]

    # Tenant B cannot read Tenant A's agent
    r_get_agent = client.get(f"/v1/agents/{agent_id}", headers=headers_b)
    assert r_get_agent.status_code == 404

    # Tenant B cannot execute Tenant A's agent
    r_run_agent = client.post(
        f"/v1/agents/{agent_id}/run",
        headers=headers_b,
        json={"query": "Translate hieroglyphs"},
    )
    assert r_run_agent.status_code == 404

    # Tenant B cannot delete Tenant A's agent
    r_del_agent = client.delete(f"/v1/agents/{agent_id}", headers=headers_b)
    assert r_del_agent.status_code == 404

    # 2. Create a team under tenant_A
    create_team_resp = client.post(
        "/v1/agents/teams",
        headers=headers_a,
        json={
            "team_id": "tenant-a-team",
            "name": "Team Alpha",
            "strategy": "sequential",
            "members": ["historical_agent"],
            "config": {"tenant_id": "tenant_A"},
        },
    )
    assert create_team_resp.status_code == 200

    # Tenant B cannot read Tenant A's team
    r_get_team = client.get("/v1/agents/teams/tenant-a-team", headers=headers_b)
    assert r_get_team.status_code == 404

    # Tenant B cannot execute Tenant A's team workflow
    r_run_team = client.post(
        "/v1/agents/teams/tenant-a-team/run",
        headers=headers_b,
        json={"query": "Execute research"},
    )
    assert r_run_team.status_code == 404

    # Tenant B cannot delete Tenant A's team
    r_del_team = client.delete("/v1/agents/teams/tenant-a-team", headers=headers_b)
    assert r_del_team.status_code == 404

    # 3. Execute team workflow under tenant_A with human approval
    run_resp = client.post(
        "/v1/agents/teams/tenant-a-team/run",
        headers=headers_a,
        json={"query": "Review artifacts", "require_human_approval": True},
    )
    assert run_resp.status_code == 200
    run_id = run_resp.json()["run_id"]

    # Tenant B cannot inspect Tenant A's execution
    r_get_exec = client.get(f"/v1/agents/executions/{run_id}", headers=headers_b)
    assert r_get_exec.status_code == 404

    # Tenant B cannot approve Tenant A's execution
    r_approve = client.post(
        f"/v1/agents/executions/{run_id}/approve",
        headers=headers_b,
        json={"approved": True, "comment": "Malicious approval"},
    )
    assert r_approve.status_code == 404

    # Tenant A CAN inspect and approve
    r_get_exec_a = client.get(f"/v1/agents/executions/{run_id}", headers=headers_a)
    assert r_get_exec_a.status_code == 200

    r_approve_a = client.post(
        f"/v1/agents/executions/{run_id}/approve",
        headers=headers_a,
        json={"approved": True, "comment": "Authorized approval"},
    )
    assert r_approve_a.status_code == 200
    assert r_approve_a.json()["status"] == "COMPLETED"

    # 4. Verify list_teams and list_executions filtering via API
    r_list_teams_b = client.get("/v1/agents/teams", headers=headers_b)
    assert r_list_teams_b.status_code == 200
    assert not any(t["team_id"] == "tenant-a-team" for t in r_list_teams_b.json())

    r_list_exec_b = client.get("/v1/agents/executions", headers=headers_b)
    assert r_list_exec_b.status_code == 200
    assert not any(e["run_id"] == run_id for e in r_list_exec_b.json())

    # 5. Clean up via Tenant A authorized deletion
    r_del_team_a = client.delete("/v1/agents/teams/tenant-a-team", headers=headers_a)
    assert r_del_team_a.status_code == 200

    r_del_agent_a = client.delete(f"/v1/agents/{agent_id}", headers=headers_a)
    assert r_del_agent_a.status_code == 200
