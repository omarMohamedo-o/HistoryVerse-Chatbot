"""Unit tests for the Multi-Agent System and AgentOrchestrator."""

import pytest

from historyverse.services.agents.critic import CriticAgent
from historyverse.services.agents.historical import HistoryVerseAgent
from historyverse.services.agents.orchestrator import AgentOrchestrator, CustomAgent
from historyverse.services.agents.researcher import ResearcherAgent
from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_historyverse_mora_agent_execution():
    llm = MockLLMProvider("mock-gemini")
    agent = HistoryVerseAgent(llm_provider=llm)

    response = await agent.run("Tell me about the Serapeum")
    assert response.agent_id == "historyverse-mora"
    assert response.agent_name == "Mora"
    assert "Classical" in response.role
    assert "Serapeum" in response.output


@pytest.mark.asyncio
async def test_historyverse_agent_name_assigned_separately():
    llm = MockLLMProvider("mock-gemini")
    agent = HistoryVerseAgent(llm_provider=llm, name="Athena")
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

    mora = HistoryVerseAgent(llm_provider=llm)
    researcher = ResearcherAgent(llm_provider=llm)
    critic = CriticAgent(llm_provider=llm)

    orchestrator.register_agent(researcher)
    orchestrator.register_agent(mora)
    orchestrator.register_agent(critic)

    # 1. Test Listing
    agents = orchestrator.list_agents()
    assert len(agents) == 3

    # 2. Test Individual Run
    ind_res = await orchestrator.run_individual("historyverse-mora", "Who was Cleopatra?")
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
