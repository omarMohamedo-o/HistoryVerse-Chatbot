"""Unit and API tests for HistoryVerse AR Mode Agent."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from historyverse.api.app import create_app
from historyverse.api.dependencies import (
    get_llm_provider,
    get_tts_provider,
)
from historyverse.config.settings import Settings
from historyverse.services.agents.historyverse_ar_mode import HistoryVerseARModeAgent
from historyverse.services.memory.agent_memory import AgentMemory
from historyverse.services.memory.manager import ConversationMemory
from tests.conftest import MockLLMProvider


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def historyverse_ar_mode_agent(mock_provider: MockLLMProvider) -> HistoryVerseARModeAgent:
    backend = ConversationMemory(Settings())
    agent_memory = AgentMemory("historyverse-ar-mode", backend=backend)
    return HistoryVerseARModeAgent(
        llm_provider=mock_provider,
        artifact_label="Statue of Serapis",
        memory=agent_memory,
    )


@pytest.mark.asyncio
async def test_historyverse_ar_mode_initialization_and_label_switch(
    historyverse_ar_mode_agent: HistoryVerseARModeAgent,
):
    assert historyverse_ar_mode_agent.artifact_label == "Statue of Serapis"
    assert historyverse_ar_mode_agent.name == "Mora"
    assert historyverse_ar_mode_agent.agent_id == "historyverse-ar-mode"
    assert "Statue of Serapis" in historyverse_ar_mode_agent.system_prompt

    historyverse_ar_mode_agent.set_artifact_label("Great Sphinx of Giza")
    assert historyverse_ar_mode_agent.artifact_label == "Great Sphinx of Giza"
    assert "Great Sphinx of Giza" in historyverse_ar_mode_agent.system_prompt


@pytest.mark.asyncio
async def test_historyverse_ar_mode_start_conversation_english_and_arabic(
    historyverse_ar_mode_agent: HistoryVerseARModeAgent,
):
    # English start
    intro_en = await historyverse_ar_mode_agent.start_conversation(
        "Pharos Lighthouse", session_id="sess-ar-1", language="en"
    )
    assert intro_en is not None
    assert len(historyverse_ar_mode_agent.memory.get_history("sess-ar-1")) == 2

    # Arabic start
    intro_ar = await historyverse_ar_mode_agent.start_conversation(
        "عمود السواري", session_id="sess-ar-2", language="ar"
    )
    assert intro_ar is not None
    assert historyverse_ar_mode_agent.artifact_label == "عمود السواري"
    assert len(historyverse_ar_mode_agent.memory.get_history("sess-ar-2")) == 2


@pytest.mark.asyncio
async def test_historyverse_ar_mode_chat_and_memory_isolation(
    historyverse_ar_mode_agent: HistoryVerseARModeAgent,
):
    session_id = "sess-serapis-chat"
    resp = await historyverse_ar_mode_agent.run(
        "Who commissioned this statue and when?",
        context={"session_id": session_id},
    )
    assert resp.output is not None
    assert resp.metadata.get("artifact_label") == "Statue of Serapis"
    assert resp.metadata.get("agent_type") == "HistoryVerseARMode"

    history = historyverse_ar_mode_agent.memory.get_history(session_id)
    assert len(history) == 2
    assert "commissioned" in history[0]["content"]


def test_api_historyverse_ar_mode_start_and_chat_routes(mock_provider: MockLLMProvider):
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    client = TestClient(app)

    # 1. Test POST /v1/historyverse-ar-mode/start
    start_resp = client.post(
        "/v1/historyverse-ar-mode/start",
        json={
            "artifact_name": "Serapeum of Alexandria",
            "session_id": "api-ar-session",
            "language": "en",
            "speak_response": False,
        },
    )
    assert start_resp.status_code == 200
    data = start_resp.json()
    assert "answer" in data
    assert data["confidence_score"] == 0.95
    assert data["audio_url"] is None

    # 2. Test POST /v1/historyverse-ar-mode/chat
    chat_resp = client.post(
        "/v1/historyverse-ar-mode/chat",
        json={
            "artifact_name": "Serapeum of Alexandria",
            "query": "Tell me about the Serapeum library branch",
            "session_id": "api-ar-session",
            "speak_response": False,
        },
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "answer" in chat_data
    assert chat_data["confidence_score"] == 0.95

    # 3. Test legacy endpoint compatibility POST /v1/ar/chat
    legacy_resp = client.post(
        "/v1/ar/chat",
        json={
            "artifact_name": "Serapeum of Alexandria",
            "query": "What was housed here?",
            "session_id": "api-ar-session-2",
            "speak_response": False,
        },
    )
    assert legacy_resp.status_code == 200
    assert "answer" in legacy_resp.json()


def test_api_historyverse_ar_mode_with_voice(mock_provider: MockLLMProvider):
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = "/tmp/synthesized_ar_guide.mp3"
    app.dependency_overrides[get_tts_provider] = lambda: mock_tts

    client = TestClient(app)

    resp = client.post(
        "/v1/historyverse-ar-mode/start",
        json={
            "artifact_name": "Rosetta Stone",
            "session_id": "api-voice-ar",
            "language": "en",
            "speak_response": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["audio_url"] == "/tmp/synthesized_ar_guide.mp3"
