"""Integration tests covering additional API endpoints: teams run/get/delete, executions inspection, video analysis, voice disabled branches, and prometheus export."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent.api.app import create_app
from agent.api.dependencies import (
    get_app_settings,
    get_file_chat_service,
    get_image_analyzer,
    get_rag_pipeline,
    get_stt_provider,
    get_tts_provider,
)
from agent.config.settings import Settings
from agent.core.schemas import ChatResponse


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_api_teams_crud_and_run(client):
    """Verify POST /v1/agents/teams/run, GET /teams/{id}, and DELETE /teams/{id}."""
    headers = {"X-API-Key": "dev-local-key"}

    # 1. Create team first
    create_resp = client.post(
        "/v1/agents/teams",
        headers=headers,
        json={
            "team_id": "test-curators",
            "name": "Historical Curators",
            "strategy": "sequential",
            "members": ["historical_agent"],
        },
    )
    assert create_resp.status_code == 200

    # 2. Get team details
    get_resp = client.get("/v1/agents/teams/test-curators", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Historical Curators"

    # 3. Run team pipeline via POST /v1/agents/team/run
    run_resp = client.post(
        "/v1/agents/team/run",
        headers=headers,
        json={"query": "Explore the Library of Alexandria", "pipeline": ["historical_agent"]},
    )
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert "final_answer" in data

    # 4. Delete team
    del_resp = client.delete("/v1/agents/teams/test-curators", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"


def test_api_execution_inspection(client):
    """Verify GET /v1/agents/executions/{run_id}."""
    headers = {"X-API-Key": "dev-local-key"}

    # Trigger a team workflow to create an execution run
    client.post(
        "/v1/agents/teams",
        headers=headers,
        json={
            "team_id": "audit-team",
            "name": "Audit Team",
            "strategy": "sequential",
            "members": ["historical_agent"],
        },
    )
    exec_resp = client.post(
        "/v1/agents/teams/audit-team/run",
        headers=headers,
        json={"query": "Chronicle the Ptolemaic dynasty"},
    )
    assert exec_resp.status_code == 200
    run_id = exec_resp.json()["run_id"]

    # Inspect the execution run
    inspect_resp = client.get(f"/v1/agents/executions/{run_id}", headers=headers)
    assert inspect_resp.status_code == 200
    assert inspect_resp.json()["run_id"] == run_id


def test_api_video_analysis(client):
    """Verify POST /v1/video processes video upload."""
    headers = {"X-API-Key": "dev-local-key"}

    mock_analyzer = MagicMock()
    mock_analyzer.analyze_video = AsyncMock(return_value="Temple built in Greco-Roman style.")
    client.app.dependency_overrides[get_image_analyzer] = lambda: mock_analyzer

    try:
        fake_video = b"FAKEMP4_VIDEO_STREAM_BYTES"
        resp = client.post(
            "/v1/video",
            headers=headers,
            params={"query": "Identify the architectural style of this monument."},
            files={"file": ("historical_temple.mp4", fake_video, "video/mp4")},
        )
        assert resp.status_code == 200
        assert "answer" in resp.json()
        assert "Greco-Roman" in resp.json()["answer"]
    finally:
        client.app.dependency_overrides.pop(get_image_analyzer, None)


def test_api_voice_disabled_branches(client):
    """Verify Voice routes return 400 when voice is disabled in configuration."""
    disabled_settings = Settings(voice_enabled=False)
    client.app.dependency_overrides[get_app_settings] = lambda: disabled_settings
    headers = {"X-API-Key": "dev-local-key"}

    try:
        # Transcribe & chat disabled
        r1 = client.post(
            "/v1/voice",
            headers=headers,
            files={"file": ("test.wav", b"RIFFbytes", "audio/wav")},
        )
        assert r1.status_code == 400

        # STS disabled
        r2 = client.post(
            "/v1/sts",
            headers=headers,
            files={"file": ("test.wav", b"RIFFbytes", "audio/wav")},
        )
        assert r2.status_code == 400
    finally:
        client.app.dependency_overrides.pop(get_app_settings, None)


def test_api_mcp_tool_execute_failure(client):
    """Verify POST /v1/mcp/tools/{name}/execute returns 400 on tool execution failure."""
    headers = {"X-API-Key": "dev-local-key"}
    resp = client.post(
        "/v1/mcp/tools/non_existent_tool/execute",
        headers=headers,
        json={"arguments": {}},
    )
    assert resp.status_code == 400
    assert "not recognized in the harness registry" in resp.json()["detail"]


def test_api_health_metrics_with_prometheus(client, monkeypatch):
    """Verify GET /metrics returns prometheus exposition when prometheus_client is available."""
    mock_prom = MagicMock()
    mock_prom.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    mock_prom.generate_latest = MagicMock(
        return_value=b"# HELP agent_requests_total\nagent_requests_total 42.0\n"
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", mock_prom)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"agent_requests_total 42.0" in resp.content


def test_api_voice_transcribe_tts_failure_branch(client):
    """Verify POST /v1/voice handles TTS synthesis failure gracefully when speak_response is True."""
    headers = {"X-API-Key": "dev-local-key"}
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "Tell me about Giza"

    mock_tts = MagicMock()
    mock_tts.synthesize.side_effect = RuntimeError("TTS audio synthesis failed")

    mock_rag = MagicMock()
    dummy_resp = ChatResponse(
        question="Tell me about Giza",
        answer="The Great Pyramid of Giza was built for Khufu.",
        model_used="test-llm",
        confidence_score=0.95,
        audio_url=None,
    )
    mock_rag.query = AsyncMock(return_value=dummy_resp)

    client.app.dependency_overrides[get_stt_provider] = lambda: mock_stt
    client.app.dependency_overrides[get_tts_provider] = lambda: mock_tts
    client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag

    try:
        resp = client.post(
            "/v1/voice?speak_response=true",
            headers=headers,
            files={"file": ("speech.wav", b"RIFFfakeaudio", "audio/wav")},
        )
        assert resp.status_code == 200
        assert "Great Pyramid" in resp.json()["answer"]
        assert resp.json().get("audio_url") is None
    finally:
        client.app.dependency_overrides.pop(get_stt_provider, None)
        client.app.dependency_overrides.pop(get_tts_provider, None)
        client.app.dependency_overrides.pop(get_rag_pipeline, None)


def test_api_chat_file_with_tts_audio_url(client):
    """Verify POST /v1/chat/file attaches audio_url when speak_response is True and TTS succeeds."""
    headers = {"X-API-Key": "dev-local-key"}
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = "/outputs/audio/file_chat.mp3"

    mock_file_chat = MagicMock()
    dummy_file_resp = ChatResponse(
        question="Summarize this papyrus",
        answer="Papyrus contains administrative records.",
        model_used="test-llm",
        confidence_score=0.95,
        audio_url=None,
    )
    mock_file_chat.chat_with_file_bytes = AsyncMock(return_value=dummy_file_resp)

    client.app.dependency_overrides[get_tts_provider] = lambda: mock_tts
    client.app.dependency_overrides[get_file_chat_service] = lambda: mock_file_chat

    try:
        resp = client.post(
            "/v1/chat/file",
            headers=headers,
            data={"query": "Summarize this papyrus", "speak_response": "true"},
            files={"file": ("papyrus.txt", b"Ancient Egyptian text content", "text/plain")},
        )
        assert resp.status_code == 200
        assert resp.json()["audio_url"] == "/outputs/audio/file_chat.mp3"
    finally:
        client.app.dependency_overrides.pop(get_tts_provider, None)
        client.app.dependency_overrides.pop(get_file_chat_service, None)


def test_api_chat_websocket_langid_and_exception(client):
    """Verify /v1/ws/chat detects language and handles unexpected pipeline exceptions."""
    mock_rag = MagicMock()
    dummy_ws_resp = ChatResponse(
        question="Bonjour",
        answer="Bonjour le monde",
        model_used="test-llm",
        confidence_score=0.95,
    )
    mock_rag.query = AsyncMock(
        side_effect=[dummy_ws_resp, RuntimeError("Unexpected pipeline crash")]
    )
    client.app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag

    mock_langid = MagicMock()
    mock_langid.classify.return_value = ("fr", 0.99)

    try:
        with (
            patch.dict("sys.modules", {"langid": mock_langid}),
            client.websocket_connect("/v1/ws/chat") as ws,
        ):
            ws.send_text("Bonjour le monde")
            data = ws.receive_json()
            assert data["answer"] == "Bonjour le monde"
            assert data["language"] == "fr"

            # Send second query to trigger exception branch
            ws.send_text("Crash query")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
    finally:
        client.app.dependency_overrides.pop(get_rag_pipeline, None)
