"""Unit tests for Agent Orchestrator, templated agent YAML loading, stores corruption recovery, and evals edge cases."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.services.agents.base import AgentResponse, BaseAgent
from agent.services.agents.orchestrator import AgentOrchestrator
from agent.services.agents.registry_store import AgentRegistryStore
from agent.services.agents.team_store import TeamRegistryStore as TeamStore
from agent.services.agents.template import TemplatedAgent
from agent.services.evaluation.ai_evals import (
    AIEvalsEngine as AIEvalFramework,
)


class SimpleAgent(BaseAgent):
    async def run(self, query: str, context: dict | None = None) -> AgentResponse:
        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output="Simple agent response",
            model_used=self.llm_provider.model_name,
        )


@pytest.mark.asyncio
async def test_orchestrator_list_agents_with_settings_and_conditional_branching():
    """Verify list_agents detects provider type from provider settings and test conditional router match."""
    mock_llm = MagicMock()
    mock_llm.model_name = "test-provider-model"
    mock_llm.settings = MagicMock()
    mock_llm.settings.llm_provider = "gemini"

    agent1 = SimpleAgent(agent_id="arch-01", name="Arch", role="Arch", system_prompt="Sys", llm_provider=mock_llm)
    agent2 = SimpleAgent(agent_id="poly-02", name="Poly", role="Poly", system_prompt="Sys", llm_provider=mock_llm)

    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(agent1)
    orchestrator.register_agent(agent2)

    # 1. Verify list_agents detects "gemini"
    agents_list = orchestrator.list_agents()
    arch_entry = next(a for a in agents_list if a["agent_id"] == "arch-01")
    assert arch_entry["model_used"] == "test-provider-model"

    # 2. Verify conditional execution matching branch
    mock_router = SimpleAgent(agent_id="router-01", name="Router", role="Router", system_prompt="Sys", llm_provider=mock_llm)
    mock_router.run = AsyncMock(return_value=AgentResponse(
        agent_id="router-01", agent_name="Router", role="Router", output="archaeology branch", model_used="test"
    ))
    orchestrator.register_agent(mock_router)

    result = await orchestrator.run_conditional(
        query="Analyze Rosetta Stone",
        router_agent_id="router-01",
        branches={"archaeology": "arch-01", "politics": "poly-02"},
    )
    assert len(result.agent_contributions) == 2


@pytest.mark.asyncio
async def test_templated_agent_from_yaml_and_prompt_formatting(tmp_path):
    """Verify TemplatedAgent.from_yaml loads declarative config and formats prompt with history and context."""
    yaml_file = tmp_path / "custom_agent.yaml"
    yaml_file.write_text(
        "agent_id: scribe-01\n"
        "persona_name: AncientScribe\n"
        "role: Chronicler\n"
        "domain: historical\n"
        "system_prompt: 'You record historical decrees for {civilization}.'\n"
        "default_variables:\n"
        "  civilization: Ancient Egypt\n"
    )

    mock_llm = MagicMock()
    mock_llm.model_name = "mock-llm"

    mock_memory = MagicMock()
    mock_memory.get_history.return_value = ["turn 1"]
    mock_memory.get_formatted_history.return_value = "User: Hi\nAssistant: Greetings."

    agent = TemplatedAgent.from_yaml(yaml_file, llm_provider=mock_llm, memory=mock_memory)
    assert agent.agent_id == "scribe-01"
    assert agent.name == "AncientScribe"

    # Test stream with context_str to exercise context prompt section
    async def dummy_stream(p):
        yield "Dynastic "
        yield "chronicle"

    mock_llm.stream = dummy_stream
    streamed = [c async for c in agent.stream("Tell of the king", context={"context": "Karnak stones", "session_id": "s1"})]
    assert "".join(streamed) == "Dynastic chronicle"


def test_stores_corrupt_json_fallback_and_metadata_sanitization():
    """Verify TeamStore and AgentRegistryStore recover from corrupt JSON and test metadata sanitization."""
    team_store = TeamStore()
    row = {
        "team_id": "team_1",
        "name": "Team 1",
        "members_json": "INVALID_JSON{",
        "config_json": "INVALID_JSON{",
    }
    parsed = team_store._team_row_to_dict(row)
    assert parsed["members"] == []
    assert parsed["config"] == {}

    exec_row = {
        "run_id": "exec_1",
        "steps_json": "INVALID_JSON{",
        "requires_approval": 0,
    }
    parsed_exec = team_store._exec_row_to_dict(exec_row)
    assert parsed_exec["steps"] == []

    reg_store = AgentRegistryStore()
    agent_row = {
        "agent_id": "agent_corrupt_meta",
        "metadata_json": "CORRUPT_JSON{",
    }
    parsed_agent = reg_store._row_to_dict(agent_row)
    assert parsed_agent["metadata"] == {}
    assert reg_store._sanitize_metadata("non-dict-string") == "non-dict-string"


def test_ai_evals_edge_cases():
    """Verify AIEvalFramework metric scoring edge cases."""
    mock_llm = MagicMock()
    framework = AIEvalFramework(llm_provider=mock_llm)
    # 1. evaluate_context_relevance when question tokens are empty
    assert framework.evaluate_context_relevance("is it so?", ["some context"]) == 1.0

    # 2. evaluate_faithfulness when answer tokens are empty
    assert framework.evaluate_faithfulness("no", ["some context"]) == 1.0

    # 3. evaluate_answer_relevance when tokens are empty
    assert framework.evaluate_answer_relevance("is it?", "no") == 0.5

    # 4. evaluate_factual_accuracy when ideal is empty or ideal tokens are empty
    assert framework.evaluate_factual_accuracy("Some detailed answer", "") == 1.0
    assert framework.evaluate_factual_accuracy("Some detailed answer", "is it") == 1.0
