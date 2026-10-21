"""Multimodal image, video, and image generation endpoints."""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, UploadFile

from agentverse.api.dependencies import (
    get_app_settings,
    get_image_analyzer,
    get_image_generator,
    get_video_generator,
)
from agentverse.config.settings import Settings
from agentverse.core.exceptions import VisionError
from agentverse.core.schemas import (
    ChatResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from agentverse.services.vision.analyzer import ImageAnalyzer
from agentverse.services.vision.generator import ImageGenerator
from agentverse.services.vision.video_generator import VideoGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.post("/vision", response_model=ChatResponse)
async def process_image_query(
    image_source: str,
    query: str,
    settings: Settings = Depends(get_app_settings),
    analyzer: ImageAnalyzer = Depends(get_image_analyzer),
) -> ChatResponse:
    """Analyze historical images or museum artifacts using visual reasoning."""
    if not settings.vision_enabled:
        raise VisionError("Vision analysis is disabled in configuration.")

    answer = await analyzer.analyze(image_source=image_source, query=query)
    return ChatResponse(
        question=query,
        answer=answer,
        sources=[image_source],
        confidence_score=0.95,
        model_used="multimodal-vision",
    )


@router.post("/video", response_model=ChatResponse)
async def process_video_query(
    query: str,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_app_settings),
    analyzer: ImageAnalyzer = Depends(get_image_analyzer),
) -> ChatResponse:
    """Analyze uploaded historical video clip using keyframe sequence reasoning."""
    if not settings.vision_enabled or not settings.video_enabled:
        raise VisionError("Video analysis is disabled in configuration.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        answer = await analyzer.analyze_video(video_source=tmp_path, query=query)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return ChatResponse(
        question=query,
        answer=answer,
        sources=[file.filename or "uploaded-video.mp4"],
        confidence_score=0.92,
        model_used="multimodal-video",
    )


@router.post("/vision/generate", response_model=ImageGenerationResponse)
async def generate_historical_image(
    req: ImageGenerationRequest,
    settings: Settings = Depends(get_app_settings),
    generator: ImageGenerator = Depends(get_image_generator),
) -> ImageGenerationResponse:
    """Synthesize a historically authentic image reconstruction."""
    if not settings.image_generation_enabled:
        raise VisionError("Image creation is disabled in application settings.")

    return await generator.generate_historical_image(
        prompt=req.prompt,
        size=req.size,
        quality=req.quality,
        enhance=req.enhance_prompt,
    )


@router.post("/vision/video/generate", response_model=VideoGenerationResponse)
async def generate_historical_video(
    req: VideoGenerationRequest,
    settings: Settings = Depends(get_app_settings),
    generator: VideoGenerator = Depends(get_video_generator),
) -> VideoGenerationResponse:
    """Synthesize a historically authentic video reconstruction or animation."""
    if not settings.video_generation_enabled or not settings.video_enabled:
        raise VisionError("Video generation is disabled in application settings.")

    return await generator.generate_historical_video(
        prompt=req.prompt,
        image_url=req.image_url,
        duration_seconds=req.duration_seconds,
        fps=req.fps,
        resolution=req.resolution,
        aspect_ratio=req.aspect_ratio,
        enhance=req.enhance_prompt,
    )
