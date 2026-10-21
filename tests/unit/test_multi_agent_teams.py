"""Unit tests for Multi-Agent Teams, Workflow Strategies, Execution Auditing, and Human Approval."""

import pytest
from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.dependencies import get_llm_provider
from agent.config.settings import Settings
from agent.services.agents.orchestrator import AgentOrchestrator
from agent.services.agents.team_store import TeamRegistryStore
from agent.services.agents.template import AgentTemplate, TemplatedAgent
from tests.conftest import MockLLMProvider


def test_team_registry_store_crud(tmp_path):
    db_file = str(tmp_path / "teams.db")
    store = TeamRegistryStore(db_path=db_file)

    # 1. Save team
    team = store.save_team(
        team_id="t-medical-council",
        name="Clinical Diagnostics Council",
        strategy="parallel",
        members=["dr-sarah", "dr-house"],
        description="Multi-specialist clinical review",
        config={"aggregator_id": "dr-sarah"},
    )
    assert team["team_id"] == "t-medical-council"
    assert team["strategy"] == "parallel"
    assert len(team["members"]) == 2

    # 2. Get team
    fetched = store.get_team("t-medical-council")
    assert fetched is not None
    assert fetched["name"] == "Clinical Diagnostics Council"

    # 3. List teams
    all_teams = store.list_teams()
    assert len(all_teams) == 1

    # 4. Create and update execution
    run = store.create_execution(
        run_id="run-101",
        team_id="t-medical-council",
        strategy="parallel",
        query="Analyze symptoms of acute appendicitis",
    )
    assert run["run_id"] == "run-101"
    assert run["status"] == "RUNNING"

    updated_run = store.update_execution(
        run_id="run-101",
        status="COMPLETED",
        final_answer="Diagnosis confirmed.",
        steps=[{"step_id": "step_1", "output": "Findings"}],
    )
    assert updated_run["status"] == "COMPLETED"
    assert updated_run["final_answer"] == "Diagnosis confirmed."
    assert len(updated_run["steps"]) == 1

    # 5. Delete team
    deleted = store.delete_team("t-medical-council")
    assert deleted is True
    assert store.get_team("t-medical-council") is None


@pytest.mark.asyncio
async def test_orchestrator_parallel_strategy(tmp_path):
    settings = Settings(agent_registry_db_path=str(tmp_path / "test.db"))
    orch = AgentOrchestrator(settings=settings)
    mock_llm = MockLLMProvider()

    t1 = AgentTemplate(
        agent_id="agent-a",
        name="Agent A",
        role="Specialist A",
        domain="history",
        system_prompt_template="You are Specialist A",
    )
    t2 = AgentTemplate(
        agent_id="agent-b",
        name="Agent B",
        role="Specialist B",
        domain="archaeology",
        system_prompt_template="You are Specialist B",
    )
    orch.register_agent(TemplatedAgent(template=t1, llm_provider=mock_llm))
    orch.register_agent(TemplatedAgent(template=t2, llm_provider=mock_llm))

    res = await orch.run_parallel(
        query="Tell me about the Serapeum",
        agent_ids=["agent-a", "agent-b"],
        aggregator_agent_id="agent-a",
    )
    assert res.final_answer != ""
    assert len(res.agent_contributions) >= 2
    assert res.metadata["strategy"] == "parallel"


@pytest.mark.asyncio
async def test_orchestrator_supervisor_strategy(tmp_path):
    settings = Settings(agent_registry_db_path=str(tmp_path / "test.db"))
    orch = AgentOrchestrator(settings=settings)
    mock_llm = MockLLMProvider()

    sup_t = AgentTemplate(
        agent_id="sup-1",
        name="Supervisor",
        role="Lead Coordinator",
        domain="management",
        system_prompt_template="You are Supervisor",
    )
    w_t = AgentTemplate(
        agent_id="worker-1",
        name="Worker",
        role="Field Researcher",
        domain="history",
        system_prompt_template="You are Worker",
    )
    orch.register_agent(TemplatedAgent(template=sup_t, llm_provider=mock_llm))
    orch.register_agent(TemplatedAgent(template=w_t, llm_provider=mock_llm))

    res = await orch.run_supervisor(
        query="Research the Lighthouse of Alexandria",
        supervisor_id="sup-1",
        worker_ids=["worker-1"],
        max_steps=2,
    )
    assert res.final_answer != ""
    assert len(res.agent_contributions) >= 2
    assert res.metadata["strategy"] == "supervisor"


@pytest.mark.asyncio
async def test_orchestrator_conditional_strategy(tmp_path):
    settings = Settings(agent_registry_db_path=str(tmp_path / "test.db"))
    orch = AgentOrchestrator(settings=settings)
    mock_llm = MockLLMProvider()

    router_t = AgentTemplate(
        agent_id="router-1",
        name="Router",
        role="Dispatcher",
        domain="routing",
        system_prompt_template="Classify into: history or science",
    )
    hist_t = AgentTemplate(
        agent_id="hist-1",
        name="History Bot",
        role="Historian",
        domain="history",
        system_prompt_template="You answer history",
    )
    orch.register_agent(TemplatedAgent(template=router_t, llm_provider=mock_llm))
    orch.register_agent(TemplatedAgent(template=hist_t, llm_provider=mock_llm))

    res = await orch.run_conditional(
        query="Explain Ptolemaic Egypt",
        router_agent_id="router-1",
        branches={"history": "hist-1"},
    )
    assert res.final_answer != ""
    assert res.metadata["strategy"] == "conditional"


@pytest.mark.asyncio
async def test_orchestrator_human_approval_workflow(tmp_path):
    settings = Settings(agent_registry_db_path=str(tmp_path / "approval.db"))
    orch = AgentOrchestrator(settings=settings)
    mock_llm = MockLLMProvider()

    a_t = AgentTemplate(
        agent_id="worker-agent",
        name="Worker",
        role="Assistant",
        domain="general",
        system_prompt_template="You assist.",
    )
    orch.register_agent(TemplatedAgent(template=a_t, llm_provider=mock_llm))

    # Register team
    orch.team_store.save_team(
        team_id="team-approval",
        name="Approval Team",
        strategy="sequential",
        members=["worker-agent"],
    )

    # 1. Run with require_human_approval=True
    run_res = await orch.run_team_workflow(
        team_id="team-approval",
        query="Deploy critical database changes",
        require_human_approval=True,
    )
    assert run_res["status"] == "WAITING_APPROVAL"
    assert run_res["approval_status"] == "PENDING"
    run_id = run_res["run_id"]

    # 2. Reject the checkpoint
    reject_res = await orch.resume_workflow(
        run_id=run_id, approved=False, user_comment="Unsafe query"
    )
    assert reject_res["status"] == "REJECTED"
    assert reject_res["approval_status"] == "REJECTED"

    # 3. New run and Approve
    run_res2 = await orch.run_team_workflow(
        team_id="team-approval",
        query="Safe summary task",
        require_human_approval=True,
    )
    run_id2 = run_res2["run_id"]
    approve_res = await orch.resume_workflow(
        run_id=run_id2, approved=True, user_comment="Approved by lead"
    )
    assert approve_res["status"] == "COMPLETED"
    assert approve_res["final_answer"] != ""


def test_api_teams_and_executions_endpoints(tmp_path):
    mock_llm = MockLLMProvider()
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    # 1. Discover providers endpoint
    prov_resp = client.get("/v1/agents/providers")
    assert prov_resp.status_code == 200
    providers = prov_resp.json()
    assert any(p["provider"] == "gemini" for p in providers)
    assert any(p["provider"] == "openai" for p in providers)
    assert any(p["provider"] == "ollama" for p in providers)

    # 2. Create custom agents for the team
    client.post(
        "/v1/agents/custom",
        json={
            "agent_id": "team-member-1",
            "name": "Member 1",
            "role": "Analyst",
            "system_prompt": "You analyze data accurately.",
        },
    )
    client.post(
        "/v1/agents/custom",
        json={
            "agent_id": "team-member-2",
            "name": "Member 2",
            "role": "Synthesizer",
            "system_prompt": "You summarize results clearly.",
        },
    )

    # 3. Create Team
    team_resp = client.post(
        "/v1/agents/teams",
        json={
            "team_id": "api-test-team",
            "name": "API Review Team",
            "strategy": "sequential",
            "members": ["team-member-1", "team-member-2"],
            "description": "Team for reviewing API designs",
        },
    )
    assert team_resp.status_code == 200
    assert team_resp.json()["team_id"] == "api-test-team"

    # 4. List Teams
    list_resp = client.get("/v1/agents/teams")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

    # 5. Execute Team Run
    run_resp = client.post(
        "/v1/agents/teams/api-test-team/run",
        json={"query": "Evaluate REST API contracts"},
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["status"] == "COMPLETED"
    assert "steps" in run_data
    assert len(run_data["steps"]) >= 2

    # 6. List Executions
    exec_resp = client.get("/v1/agents/executions")
    assert exec_resp.status_code == 200
    assert len(exec_resp.json()) >= 1

    # 7. Human Approval Checkpoint via API
    run_paused_resp = client.post(
        "/v1/agents/teams/api-test-team/run",
        json={"query": "Perform critical migration", "require_human_approval": True},
    )
    assert run_paused_resp.status_code == 200
    paused_data = run_paused_resp.json()
    assert paused_data["status"] == "WAITING_APPROVAL"
    paused_run_id = paused_data["run_id"]

    # 8. Approve Checkpoint via API
    approve_resp = client.post(
        f"/v1/agents/executions/{paused_run_id}/approve",
        json={"approved": True, "comment": "Approved by senior architect"},
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "COMPLETED"
