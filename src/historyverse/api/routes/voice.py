"""Voice transcription, speech-to-speech (STS), and conversation endpoints."""

from __future__ import annotations

import logging
import os
import tempfile
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse

from historyverse.api.dependencies import (
    get_conversation_memory,
    get_rag_pipeline,
    get_stt_provider,
    get_tts_provider,
)
from historyverse.core.schemas import ChatResponse
from historyverse.services.memory.manager import ConversationMemory
from historyverse.services.rag.pipeline import RAGPipeline
from historyverse.services.voice.stt import WhisperSTT
from historyverse.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.post("/voice", response_model=ChatResponse)
async def process_voice_query(
    file: UploadFile = File(...),
    speak_response: bool = True,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    memory: ConversationMemory = Depends(get_conversation_memory),
    stt: WhisperSTT = Depends(get_stt_provider),
    tts: GTTSTextToSpeech = Depends(get_tts_provider),
) -> ChatResponse:
    """Transcribe uploaded audio, run conversational RAG query, and return JSON response with audio URL."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        query_text = stt.transcribe(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    cid = "voice_session"
    history = memory.get_formatted_history(cid)

    response = await pipeline.query(
        question=query_text,
        history=history,
        conversation_id=cid,
    )
    memory.add_exchange(cid, query_text, response.answer)

    if speak_response:
        try:
            audio_path = tts.synthesize(response.answer)
            response.audio_url = audio_path
        except Exception as exc:
            logger.warning("TTS audio generation failed for voice query: %s", exc)

    return response


@router.post("/sts")
async def speech_to_speech(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    memory: ConversationMemory = Depends(get_conversation_memory),
    stt: WhisperSTT = Depends(get_stt_provider),
    tts: GTTSTextToSpeech = Depends(get_tts_provider),
) -> FileResponse:
    """Speech-to-Speech (STS): Audio In -> STT -> RAG -> TTS -> Audio Stream Out."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        query_text = stt.transcribe(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    cid = "sts_session"
    history = memory.get_formatted_history(cid)

    response = await pipeline.query(
        question=query_text,
        history=history,
        conversation_id=cid,
    )
    memory.add_exchange(cid, query_text, response.answer)

    audio_path = tts.synthesize(response.answer)
    headers = {
        "X-Transcript-Question": query_text.encode("ascii", "ignore").decode("ascii"),
        "X-Response-Model": response.model_used,
    }
    return FileResponse(audio_path, media_type="audio/mpeg", headers=headers)
