"""Integration tests for real-time conversational AR Voice WebSocket endpoint."""

from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.dependencies import get_llm_provider
from tests.conftest import MockLLMProvider


def test_ar_websocket_live_turn_taking_and_artifact_switching():
    mock_llm = MockLLMProvider()
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    with client.websocket_connect("/ws/ar/live") as ws:
        # 1. Initialize session with artifact
        ws.send_json(
            {
                "event": "session_init",
                "artifact_name": "Rosetta Stone",
                "language": "en",
                "session_id": "test_ar_session_001",
            }
        )
        ready_msg = ws.receive_json()
        assert ready_msg["event"] == "session_ready"
        assert ready_msg["artifact_name"] == "Rosetta Stone"
        assert ready_msg["session_id"] == "test_ar_session_001"

        # 2. Dynamic artifact switching
        ws.send_json(
            {
                "event": "artifact_update",
                "artifact_name": "Golden Mask of Tutankhamun",
            }
        )
        bound_msg = ws.receive_json()
        assert bound_msg["event"] == "artifact_bound"
        assert bound_msg["artifact_name"] == "Golden Mask of Tutankhamun"

        # 3. Conversational query turn
        ws.send_json(
            {
                "event": "text_input",
                "text": "What material is this mask made of?",
            }
        )

        events_received = []
        full_deltas = []
        turn_completed = False

        # Read streaming frames
        while not turn_completed:
            msg = ws.receive_json()
            event_type = msg.get("event")
            events_received.append(event_type)

            if event_type == "llm_delta":
                full_deltas.append(msg.get("delta", ""))
            elif event_type == "tts_audio":
                assert "audio_base64" in msg
                assert len(msg["audio_base64"]) > 0
            elif event_type == "turn_complete":
                turn_completed = True

        assert "agent_thinking" in events_received
        assert "llm_delta" in events_received
        assert "turn_complete" in events_received
        assert len(full_deltas) > 0


def test_ar_websocket_live_barge_in_interruption():
    mock_llm = MockLLMProvider()
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    with client.websocket_connect("/v1/ar/live") as ws:
        ws.send_json({"event": "session_init", "artifact_name": "Pharos Lighthouse"})
        init_resp = ws.receive_json()
        assert init_resp["event"] == "session_ready"

        # Start a turn
        ws.send_json({"event": "text_input", "text": "Tell me about the Pharos Lighthouse."})

        # Immediately trigger barge-in interruption
        ws.send_json({"event": "interrupt"})

        # Collect events until interrupted is received
        interrupted = False
        for _ in range(20):
            msg = ws.receive_json()
            if msg.get("event") == "interrupted":
                interrupted = True
                break

        assert interrupted is True

        # Send ping to confirm session returned cleanly to listening state
        ws.send_json({"event": "ping"})
        pong_msg = ws.receive_json()
        assert pong_msg["event"] == "pong"


def test_ar_routes_audio_and_websocket_edge_cases():
    import base64
    from unittest.mock import patch

    mock_llm = MockLLMProvider()
    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm

    client = TestClient(app)

    # 1. /ar/start and /ar/chat with TTS failure
    with patch(
        "agent.services.voice.tts.GTTSTextToSpeech.synthesize", side_effect=Exception("TTS Failure")
    ):
        r_start = client.post(
            "/v1/ar/start",
            json={"artifact_name": "Karnak Column", "speak_response": True},
        )
        assert r_start.status_code == 200
        assert r_start.json()["audio_url"] is None

        r_chat = client.post(
            "/v1/ar/chat",
            json={
                "artifact_name": "Karnak Column",
                "query": "How tall is it?",
                "speak_response": True,
            },
        )
        assert r_chat.status_code == 200
        assert r_chat.json()["audio_url"] is None

    # 2. WebSocket audio input and malformed frames
    with client.websocket_connect("/v1/ar/live") as ws:
        # Invalid JSON text frame
        ws.send_text("MALFORMED JSON STRING")

        # Binary frame audio PCM
        dummy_pcm = b"\x00\x01" * 1600
        ws.send_bytes(dummy_pcm)

        # Base64 audio chunk event
        b64_pcm = base64.b64encode(dummy_pcm).decode("utf-8")
        ws.send_json({"event": "audio_chunk", "data": b64_pcm})

        # Ping/pong to verify connection stays healthy
        ws.send_json({"event": "ping"})
        pong = ws.receive_json()
        assert pong["event"] == "pong"
