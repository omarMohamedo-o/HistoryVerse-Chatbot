"""Conversational query and real-time live streaming endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from historyverse.api.dependencies import (
    get_conversation_memory,
    get_llm_provider,
    get_rag_pipeline,
    get_tts_provider,
)
from historyverse.core.base import BaseLLMProvider
from historyverse.core.schemas import (
    ChatRequest,
    ChatResponse,
    StructuredHistoricalAnalysis,
)
from historyverse.services.memory.manager import ConversationMemory
from historyverse.services.rag.pipeline import RAGPipeline
from historyverse.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)
router = APIRouter()

__all__ = ["router"]


@router.post("/chat", response_model=ChatResponse)
async def process_chat(
    req: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    memory: ConversationMemory = Depends(get_conversation_memory),
    tts: GTTSTextToSpeech = Depends(get_tts_provider),
) -> ChatResponse:
    """Process a conversational query using RAG, drift protection, and history tracking."""
    cid = req.conversation_id or "default"
    history_context = memory.get_formatted_history(cid)

    kwargs = {}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature

    response = await pipeline.query(
        question=req.query,
        history=history_context,
        conversation_id=cid,
        prompt_version=req.prompt_version,
        **kwargs,
    )

    memory.add_exchange(cid, req.query, response.answer)

    if req.speak_response:
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

            history = memory.get_formatted_history(session_id)
            response = await pipeline.query(question=data, history=history)

            memory.add_exchange(session_id, data, response.answer)

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
        memory.clear(session_id)
    except Exception as exc:
        logger.error("WebSocket exception: %s", exc)
        await websocket.close()
