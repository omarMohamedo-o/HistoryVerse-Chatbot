"""Historical image and video analysis service."""

from __future__ import annotations

import io
import logging
import os

import requests
from PIL import Image

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseLLMProvider
from agent.core.exceptions import VisionError

logger = logging.getLogger(__name__)

__all__ = ["ImageAnalyzer"]


class ImageAnalyzer:
    """Handles multimodal image and video inspection for historical artifacts and sites."""

    def __init__(self, llm_provider: BaseLLMProvider, settings: Settings | None = None) -> None:
        self._llm = llm_provider
        self._settings = settings or get_settings()

    def load_image(self, image_source: str) -> Image.Image:
        """Load an image from local filesystem or URL."""
        if image_source.startswith(("http://", "https://")):
            try:
                resp = requests.get(image_source, timeout=10)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as exc:
                raise VisionError(f"Failed to fetch image from URL: {exc}") from exc

        if not os.path.exists(image_source):
            raise VisionError(f"Image file does not exist: {image_source}")

        try:
            return Image.open(image_source).convert("RGB")
        except Exception as exc:
            raise VisionError(f"Invalid image format: {exc}") from exc

    def extract_video_frames(self, video_path: str, max_frames: int = 4) -> list[Image.Image]:
        """Extract representative frames from a video for multimodal analysis."""
        if not os.path.exists(video_path):
            raise VisionError(f"Video file not found: {video_path}")

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise VisionError(f"Unable to read video frames from {video_path}")

            step = max(1, total_frames // max_frames)
            frames: list[Image.Image] = []

            for i in range(max_frames):
                frame_idx = min(total_frames - 1, i * step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb_frame))

            cap.release()
            logger.info("Extracted %d keyframes from %s", len(frames), video_path)
            return frames
        except ImportError:
            logger.warning("OpenCV (cv2) not installed. Attempting direct video query.")
            return []
        except Exception as exc:
            logger.error("Failed to extract video frames: %s", exc)
            raise VisionError(f"Video extraction error: {exc}") from exc

    async def analyze(self, image_source: str, query: str) -> str:
        """Analyze an image using the LLM's multimodal vision capabilities."""
        from agent.services.nlp.arabic import clean_arabic_text
        from agent.services.security.pii import redact_egyptian_pii

        sanitized_query = clean_arabic_text(redact_egyptian_pii(query))
        image = self.load_image(image_source)
        name = self._settings.agent_persona_name
        prompt = (
            f"You are {name}, an expert historical AI. Examine this image carefully and answer:\n"
            f"{sanitized_query}"
        )
        return await self._llm.generate(prompt, images=[image])

    async def analyze_video(self, video_source: str, query: str) -> str:
        """Analyze a historical video clip by extracting keyframes and prompting the LLM."""
        from agent.services.nlp.arabic import clean_arabic_text
        from agent.services.security.pii import redact_egyptian_pii

        sanitized_query = clean_arabic_text(redact_egyptian_pii(query))
        frames = self.extract_video_frames(video_source)
        name = self._settings.agent_persona_name
        prompt = (
            f"You are {name}, an expert historical AI. Review these sequential video frames "
            f"depicting a historical site, artifact, or event, and answer:\n"
            f"{sanitized_query}"
        )
        if frames:
            return await self._llm.generate(prompt, images=frames)
        return await self._llm.generate(f"{prompt}\n[Video source: {video_source}]")
