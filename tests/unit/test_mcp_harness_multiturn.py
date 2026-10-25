"""Unit tests for MCP Harness multi-turn tool invocation loop and error handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.services.agents.base import AgentResponse, BaseAgent
from agent.services.mcp.harness import AgentHarness
from agent.services.mcp.tools import HistoricalSearchTool


class DummyAgent(BaseAgent):
    async def run(self, query: str, context: dict | None = None) -> AgentResponse:
        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            output="Direct answer",
            model_used="test-llm",
        )


@pytest.mark.asyncio
async def test_mcp_harness_multiturn_tool_execution():
    """Verify MCP harness executes tool calls emitted by agent, accumulates scratchpad, and synthesizes final answer."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        MagicMock(
            page_content="Khufu ruled Egypt during 4th dynasty.", metadata={"source": "manuscript"}
        )
    ]
    harness = AgentHarness()
    harness.registry.register(HistoricalSearchTool(retriever=mock_retriever))

    # Agent first emits Action: search_historical_archives(query="Khufu")
    # Then finishes with final synthesis
    mock_llm = MagicMock()
    mock_llm.model_name = "test-harness-llm"
    mock_llm.generate = AsyncMock(
        side_effect=[
            'Action: search_historical_archives(query="Khufu")',
            "Pharaoh Khufu reigned for 23 years based on astronomical calculations.",
        ]
    )

    agent = DummyAgent(
        agent_id="harness-agent-1",
        name="RegnalHistorian",
        role="Chronologist",
        system_prompt="You calculate historical dates.",
        llm_provider=mock_llm,
    )

    resp = await harness.execute_agent_with_harness(
        agent=agent,
        query="How long did Khufu rule?",
        max_turns=3,
    )

    assert isinstance(resp, AgentResponse)
    assert "23 years" in resp.output
    assert "search_historical_archives" in resp.metadata["tools_invoked"]
    assert resp.metadata["harness_assisted"] is True


def test_mcp_harness_extract_tool_call_malformed_json():
    """Verify _extract_tool_call handles malformed JSON without raising."""
    harness = AgentHarness()
    # Candidate enclosed with { } that is invalid json to exercise except Exception: pass
    malformed_text = 'Use this tool: {tool: "broken", invalid json block}'
    call = harness.extract_tool_call(malformed_text)
    assert call is None


@pytest.mark.asyncio
async def test_mcp_harness_max_turns_fallback_synthesis():
    """Verify MCP harness triggers fallback synthesis when maximum turns are reached without conclusion."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        MagicMock(page_content="Excavation logs.", metadata={"source": "archive"})
    ]
    harness = AgentHarness()
    harness.registry.register(HistoricalSearchTool(retriever=mock_retriever))

    mock_llm = MagicMock()
    mock_llm.model_name = "test-harness-llm"
    mock_llm.generate = AsyncMock(
        side_effect=[
            'Action: search_historical_archives(query="Khufu 1")',
            'Action: search_historical_archives(query="Khufu 2")',
            "Synthesized final historical answer from collected observations.",
        ]
    )

    agent = DummyAgent(
        agent_id="harness-agent-max",
        name="Archaeologist",
        role="Field Researcher",
        system_prompt="You examine artifacts.",
        llm_provider=mock_llm,
    )

    resp = await harness.execute_agent_with_harness(
        agent=agent,
        query="Tell me about the monument construction.",
        max_turns=2,
    )

    assert isinstance(resp, AgentResponse)
    assert "Synthesized final historical answer" in resp.output
    assert resp.metadata["turns_taken"] == 2
    assert resp.metadata["harness_assisted"] is True
