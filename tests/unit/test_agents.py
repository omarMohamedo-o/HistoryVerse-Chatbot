"""Unit tests for the Multi-Agent System and AgentOrchestrator."""

import pytest

from agent.services.agents.critic import CriticAgent
from agent.services.agents.orchestrator import AgentOrchestrator, CustomAgent
from agent.services.agents.researcher import ResearcherAgent
from agent.services.agents.template import AgentTemplate, TemplatedAgent
from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_historical_templated_agent_execution():
    llm = MockLLMProvider("mock-gemini")
    template = AgentTemplate.from_yaml("configs/prompts/historical_guide.yaml")
    agent = TemplatedAgent(template=template, llm_provider=llm)

    response = await agent.run("Tell me about the Serapeum")
    assert response.agent_name == "Mora"
    assert "Classical" in response.role
    assert "Serapeum" in response.output


@pytest.mark.asyncio
async def test_templated_agent_name_assigned_separately():
    llm = MockLLMProvider("mock-gemini")
    template = AgentTemplate(
        agent_id="guide-athena",
        name="Athena",
        role="Classical Wisdom Guide",
        domain="history",
        system_prompt_template="Your name is {name}. You are an expert AI guide.",
    )
    agent = TemplatedAgent(template=template, llm_provider=llm)
    assert agent.name == "Athena"
    assert "Your name is Athena" in agent.system_prompt


@pytest.mark.asyncio
async def test_researcher_agent_execution():
    llm = MockLLMProvider("mock-researcher")
    agent = ResearcherAgent(llm_provider=llm)

    response = await agent.run("Find documents about Ptolemy")
    assert response.agent_id == "archival-researcher"
    assert "Archival" in response.role


@pytest.mark.asyncio
async def test_critic_agent_execution():
    llm = MockLLMProvider("mock-critic")
    agent = CriticAgent(llm_provider=llm)

    context = {
        "draft_answer": "The Serapeum was built in Alexandria.",
        "evidence": ["The Serapeum of Alexandria was built under Ptolemy III."],
    }
    response = await agent.run("Is this accurate?", context)
    assert response.agent_id == "historical-critic"
    assert "hallucination_score" in response.metadata
    assert response.confidence > 0.5


@pytest.mark.asyncio
async def test_agent_orchestrator_individual_and_team():
    orchestrator = AgentOrchestrator()
    llm = MockLLMProvider("mock-llm")

    template = AgentTemplate.from_yaml("configs/prompts/historical_guide.yaml")
    mora = TemplatedAgent(template=template, llm_provider=llm)
    researcher = ResearcherAgent(llm_provider=llm)
    critic = CriticAgent(llm_provider=llm)

    orchestrator.register_agent(researcher)
    orchestrator.register_agent(mora)
    orchestrator.register_agent(critic)

    # 1. Test Listing
    agents = orchestrator.list_agents()
    assert len(agents) == 3

    # 2. Test Individual Run
    ind_res = await orchestrator.run_individual(mora.agent_id, "Who was Cleopatra?")
    assert ind_res.agent_name == "Mora"

    # 3. Test Collaborative Team Pipeline
    team_res = await orchestrator.run_team("Explain the library of Alexandria")
    assert len(team_res.agent_contributions) == 3
    assert team_res.final_answer != ""
    assert team_res.drift_detected is False


@pytest.mark.asyncio
async def test_dynamic_custom_agent_spawn():
    orchestrator = AgentOrchestrator()
    llm = MockLLMProvider("mock-ollama")

    custom = orchestrator.create_custom_agent(
        agent_id="numismatic-expert",
        name="CoinExpert",
        role="Ancient Coin Identifier",
        system_prompt="Analyze ancient Greco-Roman coins with high precision.",
        llm_provider=llm,
    )
    assert isinstance(custom, CustomAgent)
    assert orchestrator.get_agent("numismatic-expert") is not None

    res = await orchestrator.run_individual("numismatic-expert", "Identify this bronze drachma.")
    assert res.agent_name == "CoinExpert"
    assert res.metadata.get("domain") == "general"


@pytest.mark.asyncio
async def test_domain_agnostic_agent_template_instantiation():
    llm = MockLLMProvider("mock-custom-llm")
    template = AgentTemplate(
        agent_id="epigraphy-analyst",
        name="Epigrapher",
        role="Hieroglyphic Inscription Analyst",
        domain="epigraphy",
        system_prompt_template="Your name is {name}. You are an expert in {role} for domain {domain}.",
    )
    templated_agent = TemplatedAgent(
        template=template,
        llm_provider=llm,
    )

    assert templated_agent.domain == "epigraphy"
    assert templated_agent.name == "Epigrapher"
    assert templated_agent.role == "Hieroglyphic Inscription Analyst"
    assert "Your name is Epigrapher" in templated_agent.system_prompt

    response = await templated_agent.run("Transcribe this cartouche.")
    assert response.agent_id == "epigraphy-analyst"
    assert response.metadata.get("domain") == "epigraphy"
    assert response.output != ""


@pytest.mark.asyncio
async def test_orchestrator_spawns_custom_domain_agent_with_provider_settings():
    from fastapi.testclient import TestClient

    from agent.api.app import create_app
    from agent.api.dependencies import get_llm_provider

    mock_llm = MockLLMProvider("mock-fastapi-provider")
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    client = TestClient(app)

    # Register a customized domain agent via API
    resp = client.post(
        "/v1/agents/custom",
        json={
            "agent_id": "palaeography-specialist",
            "name": "PalaeoExpert",
            "role": "Ancient Manuscript Specialist",
            "domain": "palaeography",
            "system_prompt": "Analyze papyrus scripts and handwriting chronology.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "created"
    assert data["domain"] == "palaeography"
    assert data["name"] == "PalaeoExpert"

    # Verify agent appears in /v1/agents list
    list_resp = client.get("/v1/agents")
    assert list_resp.status_code == 200
    agents_list = list_resp.json()
    found = next((a for a in agents_list if a["agent_id"] == "palaeography-specialist"), None)
    assert found is not None
    assert found["domain"] == "palaeography"

    # Run the custom domain agent
    run_resp = client.post(
        "/v1/agents/palaeography-specialist/run",
        json={"query": "Examine script ligature in Oxyrhynchus papyri."},
    )
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["agent_name"] == "PalaeoExpert"
    assert run_data["metadata"]["domain"] == "palaeography"


def test_agent_template_resolve_existing_and_fallback():
    # Resolves existing template by name
    tmpl = AgentTemplate.resolve("historical_guide")
    assert tmpl.agent_id == "agent-mora"
    assert tmpl.role == "Classical & Egyptian Historical Expert"

    # Resolves with extension
    tmpl_ext = AgentTemplate.resolve("ar_mode.yaml")
    assert tmpl_ext.agent_id == "ar-mode-agent"

    # Fallback when file doesn't exist
    fallback = AgentTemplate.resolve("non_existent_domain_template_xyz")
    assert fallback.agent_id == "agent-mora"
    assert fallback.domain == "general"


def test_templated_agent_create_helper():
    mock_llm = MockLLMProvider("mock-provider")
    agent = TemplatedAgent.create("historical_guide", llm_provider=mock_llm)
    assert agent.template.agent_id == "agent-mora"
    assert agent.name == "Mora"


@pytest.mark.asyncio
async def test_domain_agent_multi_turn_execution():
    from agent.services.agents.domain import DomainAgent

    mock_llm = MockLLMProvider()
    agent = DomainAgent(
        agent_id="doc-1",
        name="Dr. House",
        role="Diagnostic Specialist",
        system_prompt="You diagnose rare diseases.",
        llm_provider=mock_llm,
        domain="medicine",
        metadata={"hospital": "Princeton Plainsboro"},
    )

    # First turn without history
    resp1 = await agent.run(
        query="Patient has autoimmune symptoms.",
        context={"session_id": "sess_101", "context": "Lab reports pending"},
    )
    assert len(resp1.output) > 0
    assert resp1.metadata["domain"] == "medicine"
    assert resp1.metadata["hospital"] == "Princeton Plainsboro"

    # Second turn with active history
    resp2 = await agent.run(
        query="What treatments are indicated?",
        context={"session_id": "sess_101"},
    )
    assert len(resp2.output) > 0
    assert len(agent.memory.get_history("sess_101")) == 4


@pytest.mark.asyncio
async def test_researcher_and_critic_memory_recording():
    llm = MockLLMProvider("mock-provider")
    researcher = ResearcherAgent(llm_provider=llm)
    r_res = await researcher.run("Query", context={"session_id": "test_session_id"})
    assert len(r_res.output) > 0
    assert len(researcher.memory.get_history("test_session_id")) == 2

    critic = CriticAgent(llm_provider=llm)
    c_res = await critic.run(
        "Query",
        context={"session_id": "test_session_id", "draft_answer": "Draft answer"},
    )
    assert len(c_res.output) > 0
    assert len(critic.memory.get_history("test_session_id")) == 2
