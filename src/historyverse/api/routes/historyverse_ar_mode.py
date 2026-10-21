"""HistoryVerse AR Mode (Augmented Reality Artifact Guide) API routes."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from historyverse.api.dependencies import (
    get_historyverse_ar_mode_agent,
    get_tts_provider,
)
from historyverse.core.schemas import ChatResponse
from historyverse.services.agents.historyverse_ar_mode import HistoryVerseARModeAgent
from historyverse.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)

historyverse_ar_mode_router = APIRouter(
    prefix="/v1/historyverse-ar-mode",
    tags=["historyverse-ar-mode", "augmented-reality", "ar-agent"],
)
legacy_ar_router = APIRouter(prefix="/v1/ar", tags=["augmented-reality", "ar-agent"])


class HistoryVerseARModeStartRequest(BaseModel):
    artifact_name: str = Field(
        ..., description="Name of the detected statue or historical artifact"
    )
    session_id: str | None = Field(default=None, description="Optional conversational session ID")
    language: str = Field(
        default="en", description="Preferred tour guide language (e.g. 'en', 'ar')"
    )
    speak_response: bool = Field(
        default=False, description="Whether to synthesize TTS voice response"
    )


class HistoryVerseARModeChatRequest(BaseModel):
    artifact_name: str = Field(..., description="Name of the historical artifact being explored")
    query: str = Field(..., min_length=1, description="Question regarding the historical artifact")
    session_id: str | None = Field(default=None, description="Conversational session ID")
    speak_response: bool = Field(
        default=False, description="Whether to synthesize TTS voice response"
    )


@historyverse_ar_mode_router.post(
    "/start", response_model=ChatResponse, status_code=status.HTTP_200_OK
)
@legacy_ar_router.post("/start", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def start_historyverse_ar_mode_conversation(
    req: HistoryVerseARModeStartRequest,
    agent: Annotated[HistoryVerseARModeAgent, Depends(get_historyverse_ar_mode_agent)],
    tts: Annotated[GTTSTextToSpeech, Depends(get_tts_provider)],
) -> ChatResponse:
    """Initialize conversation for an AR detected statue or historical artifact."""
    intro_text = await agent.start_conversation(
        artifact_name=req.artifact_name,
        session_id=req.session_id,
        language=req.language,
    )

    audio_path = None
    if req.speak_response:
        try:
            audio_path = tts.synthesize(intro_text)
        except Exception as exc:
            logger.warning("TTS generation failed for AR start: %s", exc)

    return ChatResponse(
        question=f"Overview of {req.artifact_name}",
        answer=intro_text,
        sources=[req.artifact_name],
        confidence_score=0.95,
        model_used=agent.llm_provider.model_name,
        drift_detected=False,
        audio_url=audio_path,
    )


@historyverse_ar_mode_router.post(
    "/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK
)
@legacy_ar_router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_historyverse_ar_mode_artifact(
    req: HistoryVerseARModeChatRequest,
    agent: Annotated[HistoryVerseARModeAgent, Depends(get_historyverse_ar_mode_agent)],
    tts: Annotated[GTTSTextToSpeech, Depends(get_tts_provider)],
) -> ChatResponse:
    """Ask questions strictly scoped to the active AR statue or historical artifact."""
    agent.set_artifact_label(req.artifact_name)
    response = await agent.run(
        req.query,
        context={"session_id": req.session_id},
    )

    audio_path = None
    if req.speak_response:
        try:
            audio_path = tts.synthesize(response.output)
        except Exception as exc:
            logger.warning("TTS generation failed for AR chat: %s", exc)

    return ChatResponse(
        question=req.query,
        answer=response.output,
        sources=[req.artifact_name],
        confidence_score=response.confidence,
        model_used=response.model_used,
        drift_detected=False,
        audio_url=audio_path,
    )


__all__ = [
    "HistoryVerseARModeChatRequest",
    "HistoryVerseARModeStartRequest",
    "chat_historyverse_ar_mode_artifact",
    "historyverse_ar_mode_router",
    "legacy_ar_router",
    "start_historyverse_ar_mode_conversation",
]
