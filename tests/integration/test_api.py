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


def test_chat_endpoint_with_tts_speak_response(client: TestClient):
    from unittest.mock import patch

    with patch(
        "agent.services.voice.tts.GTTSTextToSpeech.synthesize",
        return_value="/tmp/test_speech.mp3",
    ):
        payload = {"query": "Tell me about the Serapeum of Alexandria", "speak_response": True}
        response = client.post("/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["audio_url"] == "/tmp/test_speech.mp3"


def test_chat_websocket_live_stream(client: TestClient):
    with client.websocket_connect("/v1/ws/chat") as websocket:
        websocket.send_text("Who ruled ancient Egypt during the Hellenistic era?")
        data = websocket.receive_json()
        assert "answer" in data
        assert "language" in data
        assert data["question"] == "Who ruled ancient Egypt during the Hellenistic era?"


def test_health_ready_and_metrics(client: TestClient):
    resp_ready = client.get("/health/ready")
    assert resp_ready.status_code == 200
    assert "ready" in resp_ready.json()

    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200
    assert "agent_requests_total" in resp_metrics.text or "prometheus" in resp_metrics.text


def test_drift_diagnostic_routes(client: TestClient):
    from unittest.mock import patch

    with patch("subprocess.Popen"):
        r_test = client.post("/v1/drift/test", params={"prompt": "Drift test prompt"})
        assert r_test.status_code == 200
        assert "confidence_score" in r_test.json()

        r_rebuild = client.post("/v1/drift/rebuild")
        assert r_rebuild.status_code == 200
        assert r_rebuild.json()["status"] == "triggered"

        with patch("agent.services.rag.ingest.DocumentIngestor.ingest"):
            r_reingest = client.post("/v1/drift/reingest")
            assert r_reingest.status_code == 200
            assert r_reingest.json()["status"] == "reingested"


def test_chat_file_upload_and_source(client: TestClient):
    from unittest.mock import AsyncMock, patch

    from agent.core.schemas import ChatResponse

    mock_chat_res = ChatResponse(
        question="Summarize",
        answer="The document discusses ancient dynasties.",
        model_used="mock-gemini",
        confidence_score=0.92,
        sources=[],
        audio_url=None,
    )
    with patch(
        "agent.services.rag.file_chat.FileChatService.chat_with_file_source",
        new_callable=AsyncMock,
        return_value=mock_chat_res,
    ):
        resp = client.post(
            "/v1/chat",
            json={"query": "Summarize", "file_source": "sample.pdf"},
        )
        assert resp.status_code == 200
        assert "dynasties" in resp.json()["answer"]

    with patch(
        "agent.services.rag.file_chat.FileChatService.chat_with_file_bytes",
        new_callable=AsyncMock,
        return_value=mock_chat_res,
    ):
        files = {"file": ("sample.txt", b"Mock document content", "text/plain")}
        data = {"query": "What is this file?", "speak_response": "false"}
        resp2 = client.post("/v1/chat/file", files=files, data=data)
        assert resp2.status_code == 200
        assert "dynasties" in resp2.json()["answer"]


def test_ingest_and_unlearning_api_edge_cases(client: TestClient):
    from unittest.mock import AsyncMock, patch

    from agent.core.schemas import IngestResponse

    headers = {"X-API-Key": "dev-insecure-test-key"}
    with patch("agent.services.rag.ingest.DocumentIngestor.ingest") as mock_ingest:
        mock_ingest.return_value = IngestResponse(
            documents_loaded=2, chunks_created=5, duration_seconds=0.1
        )
        resp_ing = client.post(
            "/v1/ingest", json={"source_directory": "/tmp/test"}, headers=headers
        )
        assert resp_ing.status_code == 200
        assert resp_ing.json()["documents_loaded"] == 2

    with patch(
        "agent.services.unlearning.engine.UnlearningEngine.unlearn",
        new_callable=AsyncMock,
        side_effect=Exception("Purge failed"),
    ):
        r_forget = client.post("/v1/unlearning/forget", json={"target_concept": "Myth"})
        assert r_forget.status_code == 500

    with patch(
        "agent.services.unlearning.engine.UnlearningEngine.validate_concept",
        new_callable=AsyncMock,
        side_effect=Exception("Validation failed"),
    ):
        r_val = client.post("/v1/unlearning/validate", json={"target_concept": "Myth"})
        assert r_val.status_code == 500

    r_del_tomb = client.delete("/v1/unlearning/tombstones/nonexistent_tombstone")
    assert r_del_tomb.status_code == 404
