"""Integration tests covering additional API endpoints: teams run/get/delete, executions inspection, video analysis, voice disabled branches, and prometheus export."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.dependencies import get_app_settings, get_image_analyzer
from agent.config.settings import Settings


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
    mock_prom.generate_latest = MagicMock(return_value=b"# HELP agent_requests_total\nagent_requests_total 42.0\n")
    monkeypatch.setitem(sys.modules, "prometheus_client", mock_prom)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"agent_requests_total 42.0" in resp.content
