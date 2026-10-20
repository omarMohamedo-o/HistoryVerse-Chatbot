"""Unit tests for multimodal historical vision analyzer and generator."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from historyverse.config.settings import Settings
from historyverse.core.exceptions import VisionError
from historyverse.services.vision.analyzer import ImageAnalyzer
from historyverse.services.vision.generator import ImageGenerator
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
