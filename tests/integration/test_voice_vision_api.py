"""Integration tests for multimodal voice and vision API endpoints."""

import io
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from historyverse.api.app import create_app
from historyverse.api.dependencies import (
    get_app_settings,
    get_image_analyzer,
    get_image_generator,
    get_rag_pipeline,
    get_stt_provider,
    get_tts_provider,
    get_video_generator,
)
from historyverse.config.settings import Settings
from historyverse.core.schemas import (
    ChatResponse,
    ImageGenerationResponse,
    VideoGenerationResponse,
)
from historyverse.services.rag.pipeline import RAGPipeline
from historyverse.services.vision.analyzer import ImageAnalyzer
from historyverse.services.vision.generator import ImageGenerator
from historyverse.services.vision.video_generator import VideoGenerator
from historyverse.services.voice.stt import WhisperSTT
from historyverse.services.voice.tts import GTTSTextToSpeech


@pytest.fixture
def client_with_overrides():
    app = create_app()
    settings = Settings(
        voice_enabled=True,
        stt_enabled=True,
        tts_enabled=True,
        vision_enabled=True,
        image_generation_enabled=True,
    )

    mock_stt = MagicMock(spec=WhisperSTT)
    mock_stt.transcribe.return_value = "Tell me about Cleopatra."

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fake_mp3:
        fake_mp3.write(b"ID3dummyvoicepayload")
        fake_mp3_path = fake_mp3.name

    mock_tts = MagicMock(spec=GTTSTextToSpeech)
    mock_tts.synthesize.return_value = fake_mp3_path

    mock_analyzer = MagicMock(spec=ImageAnalyzer)
    mock_analyzer.analyze = AsyncMock(
        return_value="The image depicts the Lighthouse of Alexandria under the Ptolemies."
    )

    mock_generator = MagicMock(spec=ImageGenerator)
    mock_generator.generate_historical_image = AsyncMock(
        return_value=ImageGenerationResponse(
            original_prompt="Pharos at sunset",
            enhanced_prompt="Detailed classical reconstruction of Pharos Lighthouse",
            image_url="https://historyverse.test/pharos.png",
            model_used="dall-e-3",
        )
    )

    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.query = AsyncMock(
        return_value=ChatResponse(
            question="Tell me about Cleopatra.",
            answer="Cleopatra VII Philopator was queen of Egypt.",
            sources=["history.txt"],
            confidence_score=0.95,
            model_used="gemini-1.5-flash",
        )
    )

    mock_video_generator = MagicMock(spec=VideoGenerator)
    mock_video_generator.generate_historical_video = AsyncMock(
        return_value=VideoGenerationResponse(
            original_prompt="Pharos sweeping light over the sea",
            enhanced_prompt="Cinematic drone pan over classical Alexandria Pharos",
            video_url="https://historyverse.test/pharos_pan.mp4",
            duration_seconds=4.0,
            fps=24,
            model_used="luma/dream-machine",
        )
    )

    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_stt_provider] = lambda: mock_stt
    app.dependency_overrides[get_tts_provider] = lambda: mock_tts
    app.dependency_overrides[get_image_analyzer] = lambda: mock_analyzer
    app.dependency_overrides[get_image_generator] = lambda: mock_generator
    app.dependency_overrides[get_video_generator] = lambda: mock_video_generator
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    if os.path.exists(fake_mp3_path):
        os.remove(fake_mp3_path)


def test_vision_analyze_api(client_with_overrides):
    response = client_with_overrides.post(
        "/v1/vision",
        params={"image_source": "https://example.com/artifact.jpg", "query": "What is this?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "Lighthouse of Alexandria" in data["answer"]
    assert data["confidence_score"] == 0.95


def test_vision_generate_api(client_with_overrides):
    response = client_with_overrides.post(
        "/v1/vision/generate",
        json={"prompt": "Pharos at sunset", "size": "1024x1024", "enhance_prompt": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_prompt"] == "Pharos at sunset"
    assert "pharos.png" in data["image_url"]


def test_vision_video_generate_api(client_with_overrides):
    response = client_with_overrides.post(
        "/v1/vision/video/generate",
        json={
            "prompt": "Pharos sweeping light over the sea",
            "duration_seconds": 4,
            "fps": 24,
            "enhance_prompt": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_prompt"] == "Pharos sweeping light over the sea"
    assert "pharos_pan.mp4" in data["video_url"]
    assert data["duration_seconds"] == 4.0
    assert data["fps"] == 24
    assert data["model_used"] == "luma/dream-machine"


def test_voice_transcribe_and_chat_api(client_with_overrides):
    fake_audio = io.BytesIO(b"RIFFdummydata")
    response = client_with_overrides.post(
        "/v1/voice",
        files={"file": ("test.wav", fake_audio, "audio/wav")},
        params={"speak_response": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["audio_url"] is not None


def test_voice_sts_api(client_with_overrides):
    fake_audio = io.BytesIO(b"RIFFdummydata")
    response = client_with_overrides.post(
        "/v1/sts",
        files={"file": ("input.wav", fake_audio, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert "Tell me about Cleopatra." in response.headers.get("x-transcript-question", "")
