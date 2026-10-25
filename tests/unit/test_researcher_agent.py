"""Unit tests for ResearcherAgent retrieval and query answering behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from agent.services.agents.base import AgentResponse
from agent.services.agents.researcher import ResearcherAgent


@pytest.mark.asyncio
async def test_researcher_agent_with_retriever_success():
    """Verify ResearcherAgent queries retriever and injects documents into prompt."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="The Great Pyramid of Giza was built for Pharaoh Khufu."),
        Document(page_content="Construction was completed around 2560 BC."),
    ]

    mock_llm = MagicMock()
    mock_llm.model_name = "test-researcher-model"
    mock_llm.generate = AsyncMock(return_value="Pharaoh Khufu built the Great Pyramid in 2560 BC.")

    agent = ResearcherAgent(
        agent_id="researcher-01",
        llm_provider=mock_llm,
        retriever=mock_retriever,
    )

    resp = await agent.run("Who built the Great Pyramid?")
    assert isinstance(resp, AgentResponse)
    assert resp.agent_id == "researcher-01"
    assert "Khufu" in resp.output
    mock_retriever.retrieve.assert_called_once_with("Who built the Great Pyramid?", top_k=3)
    prompt_sent = mock_llm.generate.call_args[0][0]
    assert "Evidence Documents:" in prompt_sent
    assert "Khufu" in prompt_sent


@pytest.mark.asyncio
async def test_researcher_agent_retriever_exception_handled():
    """Verify ResearcherAgent handles retriever errors gracefully by proceeding without evidence docs."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Vector index offline")

    mock_llm = MagicMock()
    mock_llm.model_name = "test-researcher-model"
    mock_llm.generate = AsyncMock(return_value="General historical synthesis without docs.")

    agent = ResearcherAgent(
        agent_id="researcher-02",
        llm_provider=mock_llm,
        retriever=mock_retriever,
    )

    resp = await agent.run("Tell me about Alexandria.")
    assert resp.output == "General historical synthesis without docs."
    prompt_sent = mock_llm.generate.call_args[0][0]
    assert "Evidence Documents:" not in prompt_sent
    assert "Alexandria" in prompt_sent
