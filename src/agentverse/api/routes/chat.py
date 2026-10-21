"""Conversational query and real-time live streaming endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect

from agentverse.api.dependencies import (
    get_conversation_memory,
    get_file_chat_service,
    get_llm_provider,
    get_rag_pipeline,
    get_tts_provider,
)
from agentverse.core.base import BaseLLMProvider
from agentverse.core.schemas import (
    ChatRequest,
    ChatResponse,
    StructuredHistoricalAnalysis,
)
from agentverse.services.memory.manager import ConversationMemory
from agentverse.services.rag.file_chat import FileChatService
from agentverse.services.rag.pipeline import RAGPipeline
from agentverse.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.post("/chat", response_model=ChatResponse)
async def process_chat(
    req: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    file_chat: FileChatService = Depends(get_file_chat_service),
    memory: ConversationMemory = Depends(get_conversation_memory),
    tts: GTTSTextToSpeech = Depends(get_tts_provider),
) -> ChatResponse:
    """Process a conversational query using RAG, ad-hoc file QA, drift protection, and history tracking."""
    cid = req.conversation_id or "default"
    history_context = await memory.get_formatted_history_async(cid)

    kwargs = {}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature

    if req.file_source:
        response = await file_chat.chat_with_file_source(
            file_source=req.file_source,
            question=req.query,
            history=history_context,
            conversation_id=cid,
            prompt_version=req.prompt_version,
            **kwargs,
        )
    else:
        response = await pipeline.query(
            question=req.query,
            history=history_context,
            conversation_id=cid,
            prompt_version=req.prompt_version,
            **kwargs,
        )

    await memory.add_exchange_async(cid, req.query, response.answer)

    if req.speak_response:
        try:
            audio_path = tts.synthesize(response.answer)
            response.audio_url = audio_path
        except Exception as exc:
            logger.warning("TTS audio generation skipped: %s", exc)

    return response


@router.post("/chat/file", response_model=ChatResponse)
async def chat_with_file(
    file: UploadFile = File(..., description="Document file (PDF, TXT, MD, DOCX, CSV) to analyze"),
    query: str = Form(..., min_length=1, description="Question about the document"),
    conversation_id: str | None = Form(None, description="Optional conversation session ID"),
    speak_response: bool = Form(False, description="Whether to synthesize audio answer"),
    temperature: float | None = Form(None, description="Model generation temperature override"),
    file_chat: FileChatService = Depends(get_file_chat_service),
    memory: ConversationMemory = Depends(get_conversation_memory),
    tts: GTTSTextToSpeech = Depends(get_tts_provider),
) -> ChatResponse:
    """Upload a document file (PDF, TXT, DOCX, etc.) and chat with it directly."""
    import os

    cid = conversation_id or f"file_{os.urandom(4).hex()}"
    history_context = await memory.get_formatted_history_async(cid)

    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    file_bytes = await file.read()
    response = await file_chat.chat_with_file_bytes(
        file_bytes=file_bytes,
        filename=file.filename or "uploaded_document",
        question=query,
        history=history_context,
        conversation_id=cid,
        **kwargs,
    )

    await memory.add_exchange_async(cid, query, response.answer)

    if speak_response:
        try:
            audio_path = tts.synthesize(response.answer)
            response.audio_url = audio_path
        except Exception as exc:
            logger.warning("TTS audio generation skipped: %s", exc)

    return response


@router.post("/chat/structured", response_model=StructuredHistoricalAnalysis)
async def process_chat_structured(
    req: ChatRequest,
    provider: BaseLLMProvider = Depends(get_llm_provider),
) -> StructuredHistoricalAnalysis:
    """Analyze historical query and return validated structured JSON schema output."""
    return await provider.generate_structured(req.query, StructuredHistoricalAnalysis)


@router.websocket("/ws/chat")
async def websocket_live_chat(
    websocket: WebSocket,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    memory: ConversationMemory = Depends(get_conversation_memory),
):
    """Real-time bi-directional live chat with automatic language detection."""
    await websocket.accept()
    session_id = f"ws_{id(websocket)}"
    logger.info("Live WebSocket session started: %s", session_id)

    try:
        while True:
            data = await websocket.receive_text()
            if not data.strip():
                continue

            detected_lang = "en"
            try:
                import langid

                detected_lang, _ = langid.classify(data)
            except Exception:
                pass

            history = await memory.get_formatted_history_async(session_id)
            response = await pipeline.query(question=data, history=history)

            await memory.add_exchange_async(session_id, data, response.answer)

            await websocket.send_json(
                {
                    "question": data,
                    "answer": response.answer,
                    "language": detected_lang,
                    "model_used": response.model_used,
                    "confidence_score": response.confidence_score,
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket session disconnected: %s", session_id)
        await memory.clear_async(session_id)
    except Exception as exc:
        logger.error("WebSocket exception: %s", exc)
        await websocket.close()
