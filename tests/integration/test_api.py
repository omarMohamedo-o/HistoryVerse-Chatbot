"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_seconds" in data


def test_drift_status_endpoint(client: TestClient):
    response = client.get("/v1/drift/status")
    assert response.status_code == 200
    data = response.json()
    assert "confidence_threshold" in data
    assert "drift_action" in data


def test_chat_endpoint_valid_request(client: TestClient):
    payload = {"query": "Tell me about the history of Alexandria", "speak_response": False}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["query"]
    assert "answer" in data
    assert isinstance(data["sources"], list)


def test_chat_endpoint_invalid_empty_query(client: TestClient):
    payload = {"query": ""}
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 422


def test_evaluate_endpoint(client: TestClient):
    payload = {
        "question": "What is the Serapeum?",
        "answer": "The Serapeum was an ancient Greek temple in Alexandria.",
        "contexts": ["The Serapeum was an ancient temple built in Alexandria."],
    }
    response = client.post("/v1/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness" in data
    assert "answer_relevancy" in data
