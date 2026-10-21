"""Unit tests for API key authentication and endpoint security."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from agent.api.app import create_app
from agent.api.dependencies import get_app_settings, get_llm_provider
from agent.config.settings import Settings
from tests.conftest import MockLLMProvider


def test_auth_disabled_by_default():
    settings = Settings(auth_enabled=False)
    mock_llm = MockLLMProvider()

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)
    resp = client.post(
        "/v1/agents/custom",
        json={
            "agent_id": "auth-test-agent",
            "name": "Tester",
            "role": "QA Agent",
            "system_prompt": "You test authentication systems.",
        },
    )
    assert resp.status_code == 200


def test_auth_enabled_validation():
    secret_key = "secure-agent-platform-key-12345"
    settings = Settings(
        auth_enabled=True,
        api_key=SecretStr(secret_key),
    )
    mock_llm = MockLLMProvider()

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    payload = {
        "agent_id": "secured-agent",
        "name": "Secured",
        "role": "Security Officer",
        "system_prompt": "You are a cyber security officer.",
    }

    # 1. Missing API Key -> 401 Unauthorized
    resp_unauth = client.post("/v1/agents/custom", json=payload)
    assert resp_unauth.status_code == 401
    assert "Missing API Key" in resp_unauth.json()["detail"]

    # 2. Invalid API Key -> 403 Forbidden
    resp_forbidden = client.post(
        "/v1/agents/custom",
        json=payload,
        headers={"X-API-Key": "invalid-wrong-key"},
    )
    assert resp_forbidden.status_code == 403
    assert "Invalid API Key" in resp_forbidden.json()["detail"]

    # 3. Valid API Key via X-API-Key header -> 200 OK
    resp_valid_header = client.post(
        "/v1/agents/custom",
        json=payload,
        headers={"X-API-Key": secret_key},
    )
    assert resp_valid_header.status_code == 200
    assert resp_valid_header.json()["status"] == "created"

    # 4. Valid API Key via Authorization Bearer header -> 200 OK
    update_payload = {"role": "Senior Security Officer"}
    resp_bearer = client.put(
        "/v1/agents/secured-agent",
        json=update_payload,
        headers={"Authorization": f"Bearer {secret_key}"},
    )
    assert resp_bearer.status_code == 200
    assert resp_bearer.json()["role"] == "Senior Security Officer"
