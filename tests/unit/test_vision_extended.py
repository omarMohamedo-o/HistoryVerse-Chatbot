"""Unit tests for Vision analyzer, image generator download, and video generator dispatch."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from pydantic import SecretStr

from agent.config.settings import Settings
from agent.core.exceptions import VisionError
from agent.services.vision.analyzer import ImageAnalyzer as VisionAnalyzer
from agent.services.vision.generator import ImageGenerator
from agent.services.vision.video_generator import VideoGenerator


def test_vision_analyzer_corrupted_image(tmp_path):
    """Verify VisionAnalyzer raises VisionError when reading a corrupted image file."""
    bad_img = tmp_path / "corrupt.jpg"
    bad_img.write_bytes(b"NOT_A_VALID_IMAGE_BYTES_12345")

    analyzer = VisionAnalyzer(llm_provider=MagicMock())
    with pytest.raises(VisionError, match="Invalid image format"):
        analyzer.load_image(str(bad_img))


def test_vision_analyzer_zero_frame_video(tmp_path, monkeypatch):
    """Verify VisionAnalyzer raises VisionError when video has zero frames."""
    fake_vid = tmp_path / "empty.mp4"
    fake_vid.write_bytes(b"fake video")

    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.get.return_value = 0  # 0 frames
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.CAP_PROP_FRAME_COUNT = 7

    monkeypatch.setitem(sys.modules, "cv2", mock_cv2)

    analyzer = VisionAnalyzer(llm_provider=MagicMock())
    with pytest.raises(VisionError, match="Unable to read video frames"):
        analyzer.extract_video_frames(str(fake_vid))


def test_vision_analyzer_cv2_missing(tmp_path, monkeypatch):
    """Verify VisionAnalyzer returns empty frames when cv2 is not installed."""
    fake_vid = tmp_path / "test.mp4"
    fake_vid.write_bytes(b"fake video")

    monkeypatch.setitem(sys.modules, "cv2", None)

    analyzer = VisionAnalyzer(llm_provider=MagicMock())
    frames = analyzer.extract_video_frames(str(fake_vid))
    assert frames == []


def test_vision_analyzer_cv2_general_exception(tmp_path, monkeypatch):
    """Verify VisionAnalyzer raises VisionError on unexpected video extraction error."""
    fake_vid = tmp_path / "test.mp4"
    fake_vid.write_bytes(b"fake video")

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.side_effect = RuntimeError("Codec decoder crash")
    monkeypatch.setitem(sys.modules, "cv2", mock_cv2)

    analyzer = VisionAnalyzer(llm_provider=MagicMock())
    with pytest.raises(VisionError, match="Video extraction error"):
        analyzer.extract_video_frames(str(fake_vid))


@pytest.mark.asyncio
@respx.mock
async def test_image_generator_download_remote_image(tmp_path):
    """Verify ImageGenerator downloads remote image and writes to local destination path."""
    dest = tmp_path / "output.png"
    remote_url = "https://images.example.com/pyramid.png"

    respx.get(remote_url).respond(
        status_code=200,
        content=b"\x89PNG\r\n\x1a\nfake_image_bytes",
    )

    settings = Settings(openai_api_key=SecretStr("test-key"))
    generator = ImageGenerator(settings)
    saved = await generator._download_image(remote_url, str(dest))
    assert saved == str(dest)
    assert dest.exists()
    assert dest.read_bytes().startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_video_generator_cloud_dispatch(tmp_path):
    """Verify VideoGenerator dispatches to cloud video API when api key is configured and fallback is off."""
    settings = Settings(
        openai_api_key=SecretStr("sk-test-video-key"),
        video_enabled=True,
        video_generation_enabled=True,
        multimodal_fallback_enabled=False,
    )
    mock_llm = MagicMock()
    mock_llm.supports_video_generation = False
    mock_llm.generate = AsyncMock(return_value="Enhanced video prompt")
    generator = VideoGenerator(llm_provider=mock_llm, settings=settings)

    with (
        patch.object(generator, "_dispatch_cloud_video_api", return_value="https://video.example.com/stream.mp4") as mock_dispatch,
        patch.object(generator, "_download_video", return_value="/tmp/saved.mp4"),
    ):
        resp = await generator.generate_historical_video("Pharaoh building Karnak temple")
        assert resp.video_url == "/tmp/saved.mp4"
        mock_dispatch.assert_called_once()


@pytest.mark.asyncio
async def test_video_generator_cloud_dispatch_default_impl():
    """Verify default _dispatch_cloud_video_api logs debug and handles exceptions cleanly."""
    settings = Settings()
    generator = VideoGenerator(llm_provider=MagicMock(), settings=settings)
    res = await generator._dispatch_cloud_video_api("Pharaoh prompt", None, 5, "test-key")
    assert res is None

    # Test exception branch in _dispatch_cloud_video_api (lines 152-154)
    bad_prompt = MagicMock()
    bad_prompt.__len__.side_effect = TypeError("Corrupted prompt string")
    res_exc = await generator._dispatch_cloud_video_api(bad_prompt, None, 5, "test-key")
    assert res_exc is None


@pytest.mark.asyncio
@respx.mock
async def test_video_generator_download_remote_video(tmp_path):
    """Verify VideoGenerator downloads remote video and persists locally."""
    dest = tmp_path / "output.mp4"
    remote_url = "https://videos.example.com/clip.mp4"

    respx.get(remote_url).respond(
        status_code=200,
        content=b"FAKE_MP4_CONTENT_12345",
    )

    settings = Settings()
    mock_llm = MagicMock()
    generator = VideoGenerator(llm_provider=mock_llm, settings=settings)
    saved = await generator._download_video(remote_url, str(dest))
    assert saved == str(dest)
    assert dest.read_bytes() == b"FAKE_MP4_CONTENT_12345"
