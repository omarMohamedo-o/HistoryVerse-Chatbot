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


@pytest.mark.asyncio
async def test_orchestrator_multi_agent_strategies_and_edge_cases(tmp_path):
    from agent.services.agents.registry_store import AgentRegistryStore

    agent_store = AgentRegistryStore(str(tmp_path / "agents.json"))
    team_store = TeamRegistryStore(str(tmp_path / "teams.json"))
    llm = MockLLMProvider("Mock multi-agent model")

    orchestrator = AgentOrchestrator(
        settings=Settings(),
        store=agent_store,
        team_store=team_store,
    )

    orchestrator.create_custom_agent(
        "worker-1", "Worker One", "Specialist 1", "history", "Prompt 1", llm_provider=llm
    )
    orchestrator.create_custom_agent(
        "worker-2", "Worker Two", "Specialist 2", "history", "Prompt 2", llm_provider=llm
    )
    orchestrator.create_custom_agent(
        "lead-sup", "Supervisor", "Lead", "history", "Prompt Sup", llm_provider=llm
    )

    with pytest.raises(ValueError, match="not registered"):
        await orchestrator.run_individual("unregistered-id", "Query")

    par_res = await orchestrator.run_parallel(
        "Compare ancient cultures",
        ["worker-1", "worker-2"],
        aggregator_agent_id="lead-sup",
    )
    assert par_res.final_answer is not None
    assert len(par_res.agent_contributions) == 3

    par_res_no_agg = await orchestrator.run_parallel(
        "Compare ancient cultures", ["worker-1", "worker-2"]
    )
    assert len(par_res_no_agg.agent_contributions) == 2

    with pytest.raises(ValueError, match="No valid agents"):
        await orchestrator.run_parallel("Query", ["nonexistent-1"])

    sup_res = await orchestrator.run_supervisor(
        "Plan expedition", "lead-sup", ["worker-1", "worker-2"], max_steps=2
    )
    assert sup_res.final_answer is not None
    assert sup_res.metadata["strategy"] == "supervisor"

    with pytest.raises(ValueError, match="Supervisor agent 'missing' not found"):
        await orchestrator.run_supervisor("Query", "missing", ["worker-1"])

    with pytest.raises(ValueError, match="No valid worker agents"):
        await orchestrator.run_supervisor("Query", "lead-sup", ["missing-worker"])

    cond_res = await orchestrator.run_conditional(
        "military history query",
        router_agent_id="lead-sup",
        branches={"military": "worker-1", "cultural": "worker-2"},
    )
    assert cond_res.metadata["strategy"] == "conditional"

    with pytest.raises(ValueError, match="Router agent 'missing' not found"):
        await orchestrator.run_conditional("Query", "missing", {"a": "worker-1"})

    updated_a1 = orchestrator.update_custom_agent(
        "worker-1",
        name="Updated Worker One",
        role="Senior Specialist",
    )
    assert updated_a1.name == "Updated Worker One"

    with pytest.raises(ValueError, match="does not exist"):
        orchestrator.update_custom_agent("nonexistent-worker", name="New Name")


def test_agent_api_routes_404_coverage():
    app = create_app()
    client = TestClient(app)
    headers = {"X-API-Key": "dev-insecure-test-key"}

    r1 = client.get("/v1/agents/nonexistent-agent-id", headers=headers)
    assert r1.status_code == 404

    r2 = client.put("/v1/agents/nonexistent-agent-id", json={"name": "New Name"}, headers=headers)
    assert r2.status_code == 404

    r3 = client.delete("/v1/agents/nonexistent-agent-id", headers=headers)
    assert r3.status_code == 404

    r4 = client.post(
        "/v1/agents/nonexistent-agent-id/run", json={"query": "Hello"}, headers=headers
    )
    assert r4.status_code == 404

    r5 = client.get("/v1/agents/teams/nonexistent-team-id", headers=headers)
    assert r5.status_code == 404

    r6 = client.delete("/v1/agents/teams/nonexistent-team-id", headers=headers)
    assert r6.status_code == 404

    r7 = client.post(
        "/v1/agents/teams/nonexistent-team-id/run",
        json={"query": "Test task"},
        headers=headers,
    )
    assert r7.status_code == 404

    r8 = client.get("/v1/agents/executions/nonexistent-run-id", headers=headers)
    assert r8.status_code == 404

    r9 = client.post(
        "/v1/agents/executions/nonexistent-run-id/approve",
        json={"approved": True},
        headers=headers,
    )
    assert r9.status_code == 404


@pytest.mark.asyncio
async def test_orchestrator_extended_branches(tmp_path):
    from unittest.mock import patch

    from agent.services.agents.orchestrator import AgentOrchestrator
    from agent.services.agents.registry_store import AgentRegistryStore
    from agent.services.agents.team_store import TeamRegistryStore
    from tests.conftest import MockLLMProvider

    mock_llm = MockLLMProvider("mock-model")
    orchestrator = AgentOrchestrator(
        store=AgentRegistryStore(db_path=str(tmp_path / "agents.db")),
        team_store=TeamRegistryStore(db_path=str(tmp_path / "teams.db")),
    )

    # 1. agent_store property
    assert orchestrator.agent_store is not None

    # 2. create custom agent with explicit provider configs
    with patch("agent.services.llm.factory.LLMFactory.create", return_value=mock_llm):
        a_openai = orchestrator.create_custom_agent(
            agent_id="custom-openai",
            name="OpenAI Agent",
            role="Specialist",
            system_prompt="Analyze with OpenAI standards.",
            provider_name="openai",
            model_name="gpt-4o",
            temperature=0.7,
            api_key="sk-test-secret-key-12345",
        )
        assert a_openai.agent_id == "custom-openai"

        a_gemini = orchestrator.create_custom_agent(
            agent_id="custom-gemini",
            name="Gemini Agent",
            role="Specialist",
            system_prompt="Analyze with Gemini standards.",
            provider_name="gemini",
            model_name="gemini-1.5-flash",
            temperature=0.2,
            api_key="goog-test-secret-key-12345",
        )
        assert a_gemini.agent_id == "custom-gemini"

    # 3. list_agents includes provider_type
    agents_list = orchestrator.list_agents()
    assert len(agents_list) >= 2
    assert any(a["agent_id"] == "custom-openai" for a in agents_list)

    # 4. rehydrate with duplicate agent (skipped) and error injection
    orchestrator.agent_store.save_agent(
        agent_id="broken-agent",
        name="Broken",
        role="Crash",
        system_prompt="Will fail",
    )
    with patch.object(
        orchestrator, "create_custom_agent", side_effect=Exception("Instantiation error")
    ):
        loaded = orchestrator.rehydrate()
        assert loaded == 0

    # 5. run_team with missing step agent (logs warning and skips)
    team_res = await orchestrator.run_team(
        "Query", agent_pipeline=["missing-agent-123", "custom-openai"]
    )
    assert team_res.final_answer is not None

    # 6. run_parallel with missing aggregator agent
    par_res = await orchestrator.run_parallel(
        "Parallel query",
        agent_ids=["custom-openai", "custom-gemini"],
        aggregator_agent_id="nonexistent-aggregator",
    )
    assert par_res.final_answer is not None

    # 7. run_supervisor with max_steps hit
    sup_res = await orchestrator.run_supervisor(
        "Supervisor query",
        supervisor_id="custom-openai",
        worker_ids=["custom-gemini"],
        max_steps=0,
    )
    assert sup_res.metadata["steps"] == 0

    # 8. run_team_workflow for parallel, supervisor, conditional, and failure
    orchestrator.team_store.save_team(
        team_id="team-par",
        name="Parallel Team",
        strategy="parallel",
        members=["custom-openai", "custom-gemini"],
    )
    exec_par = await orchestrator.run_team_workflow("team-par", "Parallel task")
    assert exec_par["status"] == "COMPLETED"

    orchestrator.team_store.save_team(
        team_id="team-sup",
        name="Supervisor Team",
        strategy="supervisor",
        members=["custom-openai", "custom-gemini"],
        config={"supervisor_id": "custom-openai"},
    )
    exec_sup = await orchestrator.run_team_workflow("team-sup", "Supervisor task")
    assert exec_sup["status"] == "COMPLETED"

    orchestrator.team_store.save_team(
        team_id="team-cond",
        name="Conditional Team",
        strategy="conditional",
        members=["custom-openai", "custom-gemini"],
        config={"router_id": "custom-openai", "branches": {"query": "custom-gemini"}},
    )
    exec_cond = await orchestrator.run_team_workflow("team-cond", "Conditional query")
    assert exec_cond["status"] == "COMPLETED"

    # 9. run_team_workflow error recovery
    with patch.object(orchestrator, "run_team", side_effect=RuntimeError("Workflow crash")):
        orchestrator.team_store.save_team(
            team_id="team-fail",
            name="Failing Team",
            strategy="sequential",
            members=["custom-openai"],
        )
        with pytest.raises(RuntimeError, match="Workflow crash"):
            await orchestrator.run_team_workflow("team-fail", "Will crash")
