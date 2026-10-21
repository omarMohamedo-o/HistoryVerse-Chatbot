"""Unit tests for historical video generator and animation service."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentverse.config.settings import Settings
from agentverse.core.base import BaseLLMProvider
from agentverse.core.exceptions import VisionError
from agentverse.services.vision.video_generator import VideoGenerator


class MockVideoLLM(BaseLLMProvider):
    def __init__(self, can_generate_video: bool = False, model_name: str = "mock-video-llm"):
        self._can_generate_video = can_generate_video
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def is_available(self) -> bool:
        return True

    @property
    def supports_video_generation(self) -> bool:
        return self._can_generate_video

    async def generate_video(self, prompt: str, **kwargs) -> str | None:
        return "https://historyverse.test/cinematic_reconstruction.mp4"

    async def generate(self, prompt: str, **kwargs) -> str:
        return f"Cinematic aerial camera gliding over ancient marble colonnades: {prompt}"


@pytest.mark.asyncio
async def test_video_generator_disabled_raises():
    settings = Settings(video_generation_enabled=False)
    llm = MockVideoLLM()
    generator = VideoGenerator(llm, settings=settings)

    with pytest.raises(VisionError, match="Video generation is disabled"):
        await generator.generate_historical_video("Pharos Lighthouse beacon at night")


@pytest.mark.asyncio
async def test_video_generator_native_provider_support(tmp_path):
    settings = Settings(video_generation_enabled=True, video_enabled=True)
    llm = MockVideoLLM(can_generate_video=True)
    generator = VideoGenerator(llm, settings=settings)
    generator._outputs_dir = str(tmp_path)

    with patch.object(
        generator, "_download_video", AsyncMock(return_value=str(tmp_path / "video.mp4"))
    ):
        res = await generator.generate_historical_video(
            "Pharos Lighthouse beacon at night",
            duration_seconds=5,
            fps=30,
            enhance=False,
        )
        assert res.original_prompt == "Pharos Lighthouse beacon at night"
        assert res.duration_seconds == 5.0
        assert res.fps == 30
        assert "video.mp4" in res.video_url
        assert res.model_used == "mock-video-llm"


@pytest.mark.asyncio
async def test_video_generator_prompt_enhancement(tmp_path):
    settings = Settings(video_generation_enabled=True, video_enabled=True)
    llm = MockVideoLLM(can_generate_video=False)
    generator = VideoGenerator(llm, settings=settings)
    generator._outputs_dir = str(tmp_path)

    res = await generator.generate_historical_video(
        "Ancient Alexandria harbor ships",
        duration_seconds=4,
        fps=24,
        enhance=True,
    )
    assert res.original_prompt == "Ancient Alexandria harbor ships"
    assert "Cinematic aerial camera" in res.enhanced_prompt
    assert os.path.exists(res.video_url)


@pytest.mark.asyncio
async def test_video_generator_prompt_enhancement_failure_fallback(tmp_path):
    settings = Settings(video_generation_enabled=True, video_enabled=True)
    llm = MockVideoLLM(can_generate_video=False)
    llm.generate = AsyncMock(side_effect=Exception("Prompt enhancement unavailable"))
    generator = VideoGenerator(llm, settings=settings)
    generator._outputs_dir = str(tmp_path)

    res = await generator.generate_historical_video(
        "Roman triumphal arch ceremony",
        duration_seconds=3,
        fps=24,
        enhance=True,
    )
    assert res.original_prompt == "Roman triumphal arch ceremony"
    assert res.enhanced_prompt == "Roman triumphal arch ceremony"
    assert os.path.exists(res.video_url)


def test_synthesize_local_video_with_opencv_mock(tmp_path):
    llm = MockVideoLLM()
    generator = VideoGenerator(llm)

    target_file = str(tmp_path / "mock_opencv.mp4")

    mock_cv2 = MagicMock()
    mock_out = MagicMock()
    mock_cv2.VideoWriter.return_value = mock_out
    mock_cv2.VideoWriter_fourcc.return_value = 1234

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        path = generator._synthesize_local_video(
            prompt="Temple of Serapis",
            target_path=target_file,
            duration=2,
            fps=10,
        )
        assert path == target_file
        assert mock_out.release.called


def test_synthesize_local_video_fallback_without_opencv(tmp_path):
    llm = MockVideoLLM()
    generator = VideoGenerator(llm)

    target_file = str(tmp_path / "fallback_container.mp4")

    with patch.dict("sys.modules", {"cv2": None}):
        path = generator._synthesize_local_video(
            prompt="Rosetta Stone inscribe animation",
            target_path=target_file,
            duration=2,
            fps=10,
        )
        assert path == target_file
        assert os.path.exists(target_file)
        with open(target_file, "rb") as f:
            content = f.read()
        assert b"ftyp" in content
        assert b"Rosetta Stone" in content


@pytest.mark.asyncio
async def test_video_generator_download_helpers(tmp_path):
    llm = MockVideoLLM()
    generator = VideoGenerator(llm)

    # Local file returns as is
    res_local = await generator._download_video("/tmp/local_file.mp4", "/tmp/any.mp4")
    assert res_local == "/tmp/local_file.mp4"

    # Remote URL with download error returns remote URL
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=Exception("Timeout"))):
        res_remote = await generator._download_video(
            "https://cdn.test/video.mp4", str(tmp_path / "target.mp4")
        )
        assert res_remote == "https://cdn.test/video.mp4"
