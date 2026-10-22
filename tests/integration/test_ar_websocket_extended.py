"""Integration tests for AR WebSocket covering binary audio frames, base64 audio chunks, remainder audio flushes, and error events."""

import base64
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from agent.api.app import create_app
from agent.api.dependencies import (
    get_llm_provider,
    get_streaming_stt_provider,
    get_streaming_tts_provider,
    reset_dependencies_cache,
)
from tests.conftest import MockLLMProvider


def test_ar_websocket_binary_audio_and_json_audio_chunks():
    """Verify AR live WebSocket handles binary PCM frames and JSON base64 audio_chunk events."""
    reset_dependencies_cache()
    mock_llm = MockLLMProvider()

    mock_stt = MagicMock()
    mock_stt.feed_audio_chunk = AsyncMock(side_effect=[
        ("[Processing audio...]", False),
        ("", True),
        ("[Processing base64...]", False),
        ("", True),
    ])
    mock_stt.finish = AsyncMock(side_effect=["Who was Cleopatra?", "Where is Alexandria?"])

    mock_tts = MagicMock()
    mock_tts.synthesize_chunk = AsyncMock(return_value=b"MOCK_TTS_CHUNK")

    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_streaming_stt_provider] = lambda: mock_stt
    app.dependency_overrides[get_streaming_tts_provider] = lambda: mock_tts

    client = TestClient(app)

    with client.websocket_connect("/v1/ar/live") as ws:
        # 1. Initialize session with artifact
        ws.send_json(
            {
                "event": "session_init",
                "artifact_name": "Rosetta Stone",
                "language": "en",
                "session_id": "test_ar_session_002",
            }
        )
        msg1 = ws.receive_json()
        assert msg1["event"] == "session_ready"

        # 2. Send binary PCM frame
        ws.send_bytes(b"\x00\x01" * 100)
        partial_msg = ws.receive_json()
        assert partial_msg["event"] == "transcript_partial"

        # 3. Send second binary frame that triggers is_final
        ws.send_bytes(b"\x00\x00" * 100)
        final_msg = ws.receive_json()
        assert final_msg["event"] == "transcript_final"
        assert final_msg["text"] == "Who was Cleopatra?"

        # Consume the spawned turn events until turn_complete
        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt["event"])
            if evt["event"] == "turn_complete":
                break
        assert "turn_complete" in events

        # 4. Now test JSON event audio_chunk with base64
        b64_audio = base64.b64encode(b"\x00\x02" * 50).decode("ascii")
        ws.send_json({"event": "audio_chunk", "data": b64_audio})
        p_msg = ws.receive_json()
        assert p_msg["event"] == "transcript_partial"

        ws.send_json({"event": "audio_chunk", "data": b64_audio})
        f_msg = ws.receive_json()
        assert f_msg["event"] == "transcript_final"
        assert f_msg["text"] == "Where is Alexandria?"

        # Wait for turn completion
        while True:
            evt = ws.receive_json()
            if evt["event"] == "turn_complete":
                break


def test_ar_websocket_turn_error_event():
    """Verify AR WebSocket sends error event when an unexpected exception occurs during turn generation."""
    reset_dependencies_cache()
    mock_llm = MockLLMProvider()

    mock_tts = MagicMock()
    mock_tts.synthesize_chunk = AsyncMock(side_effect=RuntimeError("TTS engine crash"))

    app = create_app()
    app.dependency_overrides[get_llm_provider] = lambda: mock_llm
    app.dependency_overrides[get_streaming_tts_provider] = lambda: mock_tts

    client = TestClient(app)

    with client.websocket_connect("/v1/ar/live") as ws:
        # Initialize
        ws.send_json({"event": "session_init", "artifact_name": "Sphinx"})
        ws.receive_json()  # session_ready

        # Send text query that will fail during TTS
        ws.send_json({"event": "text_input", "text": "Describe the Sphinx."})

        # Look for error event
        error_found = False
        while True:
            evt = ws.receive_json()
            if evt["event"] == "error":
                assert "TTS engine crash" in evt["message"]
                error_found = True
                break
        assert error_found is True
