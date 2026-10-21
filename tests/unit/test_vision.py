"""Unit tests for multimodal historical vision analyzer and generator."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agent.config.settings import Settings
from agent.core.exceptions import VisionError
from agent.services.vision.analyzer import ImageAnalyzer
from agent.services.vision.generator import ImageGenerator
from tests.conftest import MockLLMProvider


def _create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ImageCapableMockLLM(MockLLMProvider):
    @property
    def supports_image_generation(self) -> bool:
        return True


def test_load_image_from_file(tmp_path):
    img_path = tmp_path / "artifact.png"
    img_path.write_bytes(_create_test_image_bytes())

    analyzer = ImageAnalyzer(MockLLMProvider())
    loaded = analyzer.load_image(str(img_path))
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (64, 64)


def test_load_image_nonexistent_file_raises():
    analyzer = ImageAnalyzer(MockLLMProvider())
    with pytest.raises(VisionError, match="Image file does not exist"):
        analyzer.load_image("/nonexistent/artifact.png")


def test_load_image_from_url():
    analyzer = ImageAnalyzer(MockLLMProvider())
    test_bytes = _create_test_image_bytes()

    mock_resp = MagicMock()
    mock_resp.content = test_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        loaded = analyzer.load_image("https://historyverse.test/papyrus.png")
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (64, 64)


def test_load_image_from_url_failure_raises():
    analyzer = ImageAnalyzer(MockLLMProvider())

    with (
        patch("requests.get", side_effect=Exception("Connection refused")),
        pytest.raises(VisionError, match="Failed to fetch image from URL"),
    ):
        analyzer.load_image("https://historyverse.test/missing.png")


@pytest.mark.asyncio
async def test_analyze_image_workflow(tmp_path):
    img_path = tmp_path / "coin.png"
    img_path.write_bytes(_create_test_image_bytes())

    llm = MockLLMProvider()
    analyzer = ImageAnalyzer(llm)

    analysis = await analyzer.analyze(str(img_path), query="Identify this coin.")
    assert "Greco-Roman Museum" in analysis


@pytest.mark.asyncio
async def test_image_generator_disabled_raises():
    settings = Settings(image_generation_enabled=False)
    generator = ImageGenerator(MockLLMProvider(), settings=settings)

    with pytest.raises(VisionError, match="Image creation is disabled"):
        await generator.generate_historical_image("Reconstruct the Pharos Lighthouse")


@pytest.mark.asyncio
async def test_image_generator_direct_provider_support(tmp_path):
    settings = Settings(image_generation_enabled=True)
    llm = ImageCapableMockLLM()
    llm.generate_image = AsyncMock(return_value="https://cdn.historyverse.test/lighthouse.png")

    generator = ImageGenerator(llm, settings=settings)
    generator._outputs_dir = str(tmp_path)

    with patch.object(
        generator, "_download_image", AsyncMock(return_value=str(tmp_path / "local.png"))
    ):
        resp = await generator.generate_historical_image("Pharos Lighthouse at sunset")
        assert resp.original_prompt == "Pharos Lighthouse at sunset"
        assert resp.image_url == str(tmp_path / "local.png")


@pytest.mark.asyncio
async def test_image_generator_litellm_fallback(tmp_path):
    settings = Settings(image_generation_enabled=True)
    llm = MockLLMProvider()

    generator = ImageGenerator(llm, settings=settings)
    generator._outputs_dir = str(tmp_path)

    mock_litellm_response = MagicMock()
    mock_data = MagicMock()
    mock_data.url = "https://cdn.openai.test/dall-e-reconstruction.png"
    mock_litellm_response.data = [mock_data]

    with (
        patch("litellm.aimage_generation", AsyncMock(return_value=mock_litellm_response)),
        patch.object(
            generator,
            "_download_image",
            AsyncMock(return_value=str(tmp_path / "reconstructed.png")),
        ),
    ):
        resp = await generator.generate_historical_image("Ancient Colosseum in Rome")
        assert "Colosseum" in resp.original_prompt
        assert resp.image_url == str(tmp_path / "reconstructed.png")


@pytest.mark.asyncio
async def test_image_generator_prompt_enhancement_fallback(tmp_path):
    settings = Settings(image_generation_enabled=True)
    llm = MockLLMProvider()
    llm.generate = AsyncMock(side_effect=Exception("Enhancement timeout"))

    generator = ImageGenerator(llm, settings=settings)
    generator._outputs_dir = str(tmp_path)

    mock_litellm_response = MagicMock()
    mock_data = MagicMock()
    mock_data.url = "https://cdn.openai.test/fallback.png"
    mock_litellm_response.data = [mock_data]

    with (
        patch("litellm.aimage_generation", AsyncMock(return_value=mock_litellm_response)),
        patch.object(
            generator,
            "_download_image",
            AsyncMock(return_value="https://cdn.openai.test/fallback.png"),
        ),
    ):
        resp = await generator.generate_historical_image("Simple Prompt", enhance=True)
        assert resp.original_prompt == "Simple Prompt"


@pytest.mark.asyncio
async def test_image_generator_litellm_failure_raises():
    settings = Settings(image_generation_enabled=True)
    llm = MockLLMProvider()
    generator = ImageGenerator(llm, settings=settings)

    with (
        patch("litellm.aimage_generation", AsyncMock(side_effect=Exception("Rate limit reached"))),
        pytest.raises(VisionError, match="Image generation failed"),
    ):
        await generator.generate_historical_image("Athena Parthenos", enhance=False)


@pytest.mark.asyncio
async def test_image_generator_download_image_local_and_error(tmp_path):
    generator = ImageGenerator(MockLLMProvider())
    # Non-http url returns as-is
    res_local = await generator._download_image(
        "file:///local/path.png", str(tmp_path / "target.png")
    )
    assert res_local == "file:///local/path.png"

    # Exception during download returns url fallback
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=Exception("Network down"))):
        res_fallback = await generator._download_image(
            "https://cdn.test/pic.png", str(tmp_path / "pic.png")
        )
        assert res_fallback == "https://cdn.test/pic.png"


def test_extract_video_frames_missing_raises():
    analyzer = ImageAnalyzer(MockLLMProvider())
    with pytest.raises(VisionError, match="Video file not found"):
        analyzer.extract_video_frames("/nonexistent/video.mp4")


def test_extract_video_frames_mock(tmp_path):
    import numpy as np

    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"dummy_mp4_bytes")

    analyzer = ImageAnalyzer(MockLLMProvider())

    dummy_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.get.return_value = 10.0  # 10 frames
    mock_cap.read.return_value = (True, dummy_frame)

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.cvtColor = MagicMock(return_value=dummy_frame)

    import sys

    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        frames = analyzer.extract_video_frames(str(video_path), max_frames=2)
        assert len(frames) >= 1


@pytest.mark.asyncio
async def test_analyze_video_workflow(tmp_path):
    video_path = tmp_path / "historic.mp4"
    video_path.write_bytes(b"dummy_bytes")

    llm = MockLLMProvider()
    analyzer = ImageAnalyzer(llm)

    # When frames are extracted
    with patch.object(analyzer, "extract_video_frames", return_value=[Image.new("RGB", (32, 32))]):
        analysis = await analyzer.analyze_video(str(video_path), query="Describe this site.")
        assert "Greco-Roman Museum" in analysis

    # When no frames extracted
    with patch.object(analyzer, "extract_video_frames", return_value=[]):
        analysis_empty = await analyzer.analyze_video(str(video_path), query="Describe this site.")
        assert "Greco-Roman Museum" in analysis_empty
