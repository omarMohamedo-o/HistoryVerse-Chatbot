"""Augmented Reality Artifact Guide API routes and Real-Time Conversational Voice WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from agent.api.dependencies import (
    get_app_settings,
    get_historyverse_ar_mode_agent,
    get_streaming_stt_provider,
    get_streaming_tts_provider,
    get_tts_provider,
)
from agent.config.settings import Settings
from agent.core.schemas import ChatResponse
from agent.services.agents.template import TemplatedAgent
from agent.services.monitoring.metrics import (
    VOICE_BARGE_IN_EVENTS,
    VOICE_SESSIONS_ACTIVE,
    VOICE_TTFA,
)
from agent.services.voice.streaming_stt import IncrementalSTT
from agent.services.voice.streaming_tts import SentenceChunker, StreamingTTS
from agent.services.voice.tts import GTTSTextToSpeech

logger = logging.getLogger(__name__)

ar_router = APIRouter(prefix="/v1/ar", tags=["augmented-reality", "ar-agent"])
legacy_ar_router = APIRouter(
    prefix="/v1/historyverse-ar-mode",
    tags=["augmented-reality", "ar-agent", "historyverse-ar-mode"],
)
historyverse_ar_mode_router = legacy_ar_router


class ARStartRequest(BaseModel):
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


class ARChatRequest(BaseModel):
    artifact_name: str = Field(..., description="Name of the historical artifact being explored")
    query: str = Field(..., min_length=1, description="Question regarding the historical artifact")
    session_id: str | None = Field(default=None, description="Conversational session ID")
    speak_response: bool = Field(
        default=False, description="Whether to synthesize TTS voice response"
    )


HistoryVerseARModeStartRequest = ARStartRequest
HistoryVerseARModeChatRequest = ARChatRequest


@ar_router.post("/start", response_model=ChatResponse, status_code=status.HTTP_200_OK)
@legacy_ar_router.post("/start", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def start_ar_conversation(
    req: ARStartRequest,
    agent: Annotated[TemplatedAgent, Depends(get_historyverse_ar_mode_agent)],
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


@ar_router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
@legacy_ar_router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_ar_artifact(
    req: ARChatRequest,
    agent: Annotated[TemplatedAgent, Depends(get_historyverse_ar_mode_agent)],
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


class LiveARVoiceSession:
    """Full-duplex real-time AR voice interaction session with turn-taking and barge-in."""

    def __init__(
        self,
        websocket: WebSocket,
        agent: TemplatedAgent,
        stt: IncrementalSTT,
        tts: StreamingTTS,
        session_id: str | None = None,
    ) -> None:
        self.ws = websocket
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self.session_id = session_id or f"ar_live_{os.urandom(4).hex()}"
        self.artifact_name = "Ancient Artifact"
        self.language = "en"
        self.state = "IDLE"  # IDLE, LISTENING, PROCESSING, SPEAKING, INTERRUPTED
        self.active_turn_task: asyncio.Task | None = None

    async def interrupt(self) -> None:
        """Cancel ongoing model generation and audio streaming on user barge-in."""
        if self.active_turn_task and not self.active_turn_task.done():
            self.active_turn_task.cancel()
            self.active_turn_task = None
            VOICE_BARGE_IN_EVENTS.inc()
            logger.info("Barge-in interruption triggered for AR session: %s", self.session_id)
        self.state = "INTERRUPTED"
        await self.ws.send_json({"event": "interrupted"})
        await self.stt.reset()
        self.state = "LISTENING"

    async def run(self) -> None:
        """Main event loop receiving text control events and binary audio chunks."""
        await self.ws.accept()
        VOICE_SESSIONS_ACTIVE.inc()
        self.state = "LISTENING"
        logger.info("AR Live Voice WebSocket connected: %s", self.session_id)

        try:
            while True:
                msg = await self.ws.receive()
                msg_type = msg.get("type")

                if msg_type == "websocket.disconnect":
                    break

                # Binary frame: Audio PCM chunk
                if msg.get("bytes"):
                    audio_chunk = msg["bytes"]
                    if self.state in ("PROCESSING", "SPEAKING"):
                        await self.interrupt()

                    partial, is_final = await self.stt.feed_audio_chunk(audio_chunk)
                    if partial:
                        await self.ws.send_json({"event": "transcript_partial", "text": partial})

                    if is_final:
                        final_text = await self.stt.finish()
                        if final_text and final_text.strip():
                            await self.ws.send_json(
                                {"event": "transcript_final", "text": final_text}
                            )
                            self._spawn_turn_task(final_text)

                # Text frame: JSON control command
                elif msg.get("text"):
                    try:
                        payload = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        continue

                    event = payload.get("event")

                    if event == "session_init":
                        self.artifact_name = payload.get("artifact_name", self.artifact_name)
                        self.language = payload.get("language", "en")
                        if payload.get("session_id"):
                            self.session_id = payload["session_id"]
                        self.agent.set_artifact_label(self.artifact_name)
                        await self.ws.send_json(
                            {
                                "event": "session_ready",
                                "session_id": self.session_id,
                                "artifact_name": self.artifact_name,
                                "language": self.language,
                            }
                        )

                    elif event == "artifact_update":
                        new_art = payload.get("artifact_name", "")
                        if new_art:
                            self.artifact_name = new_art
                            self.agent.set_artifact_label(new_art)
                            await self.ws.send_json(
                                {
                                    "event": "artifact_bound",
                                    "artifact_name": self.artifact_name,
                                }
                            )

                    elif event == "interrupt":
                        await self.interrupt()

                    elif event == "text_input":
                        query = payload.get("text", "")
                        if query:
                            if self.state in ("PROCESSING", "SPEAKING"):
                                await self.interrupt()
                            self._spawn_turn_task(query)

                    elif event == "audio_chunk":
                        b64_data = payload.get("data", "")
                        if b64_data:
                            audio_bytes = base64.b64decode(b64_data)
                            if self.state in ("PROCESSING", "SPEAKING"):
                                await self.interrupt()
                            partial, is_final = await self.stt.feed_audio_chunk(audio_bytes)
                            if partial:
                                await self.ws.send_json(
                                    {"event": "transcript_partial", "text": partial}
                                )
                            if is_final:
                                final_text = await self.stt.finish()
                                if final_text and final_text.strip():
                                    await self.ws.send_json(
                                        {"event": "transcript_final", "text": final_text}
                                    )
                                    self._spawn_turn_task(final_text)

                    elif event == "ping":
                        await self.ws.send_json({"event": "pong"})

        except WebSocketDisconnect:
            logger.info("AR Live WebSocket disconnected: %s", self.session_id)
        except Exception as exc:
            logger.error("AR Live WebSocket unexpected error: %s", exc)
        finally:
            if self.active_turn_task and not self.active_turn_task.done():
                self.active_turn_task.cancel()
            VOICE_SESSIONS_ACTIVE.dec()

    def _spawn_turn_task(self, query: str) -> None:
        if self.active_turn_task and not self.active_turn_task.done():
            self.active_turn_task.cancel()
        self.active_turn_task = asyncio.create_task(self._execute_turn(query))

    async def _execute_turn(self, query: str) -> None:
        self.state = "PROCESSING"
        t_start = time.perf_counter()
        first_audio_sent = False
        chunker = SentenceChunker()

        await self.ws.send_json({"event": "agent_thinking"})

        try:
            self.state = "SPEAKING"
            accumulated_tokens: list[str] = []
            async for token in self.agent.stream(
                query,
                context={"session_id": self.session_id, "artifact": self.artifact_name},
            ):
                await asyncio.sleep(0.01)
                accumulated_tokens.append(token)
                await self.ws.send_json({"event": "llm_delta", "delta": token})

                # Stream out sentence-chunked audio chunks
                for sentence in chunker.feed(token):
                    audio_bytes = await self.tts.synthesize_chunk(sentence)
                    if audio_bytes:
                        if not first_audio_sent:
                            ttfa = time.perf_counter() - t_start
                            VOICE_TTFA.observe(ttfa)
                            first_audio_sent = True

                        b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                        await self.ws.send_json(
                            {
                                "event": "tts_audio",
                                "audio_base64": b64_audio,
                                "text": sentence,
                            }
                        )

            # Flush any remaining text in chunker
            remainder = chunker.flush()
            if remainder:
                audio_bytes = await self.tts.synthesize_chunk(remainder)
                if audio_bytes:
                    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                    await self.ws.send_json(
                        {
                            "event": "tts_audio",
                            "audio_base64": b64_audio,
                            "text": remainder,
                        }
                    )

            full_answer = "".join(accumulated_tokens)
            await self.ws.send_json({"event": "turn_complete", "full_text": full_answer})
            self.state = "LISTENING"

        except asyncio.CancelledError:
            logger.info("Turn task cancelled (barge-in)")
            raise
        except Exception as exc:
            logger.error("Error executing AR turn: %s", exc)
            await self.ws.send_json({"event": "error", "message": str(exc)})
            self.state = "LISTENING"


async def _handle_ar_live_ws(
    websocket: WebSocket,
    agent: TemplatedAgent,
    stt: IncrementalSTT,
    tts: StreamingTTS,
    settings: Settings,
) -> None:
    if settings.auth_enabled:
        token = (
            websocket.query_params.get("api_key")
            or websocket.query_params.get("token")
            or websocket.headers.get("x-api-key")
        )
        expected = settings.api_key.get_secret_value() if settings.api_key else ""
        if not token or token != expected:
            await websocket.close(code=1008, reason="Unauthorized: Invalid API key")
            return

    session = LiveARVoiceSession(
        websocket=websocket,
        agent=agent,
        stt=stt,
        tts=tts,
    )
    await session.run()


@ar_router.websocket("/live")
@ar_router.websocket("/ws/live")
async def ar_live_websocket(
    websocket: WebSocket,
    agent: TemplatedAgent = Depends(get_historyverse_ar_mode_agent),
    stt: IncrementalSTT = Depends(get_streaming_stt_provider),
    tts: StreamingTTS = Depends(get_streaming_tts_provider),
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Real-time bi-directional conversational voice WebSocket for HistoryVerse AR."""
    await _handle_ar_live_ws(websocket, agent, stt, tts, settings)


start_historyverse_ar_mode_conversation = start_ar_conversation
chat_historyverse_ar_mode_artifact = chat_ar_artifact

__all__ = [
    "ARChatRequest",
    "ARStartRequest",
    "HistoryVerseARModeChatRequest",
    "HistoryVerseARModeStartRequest",
    "LiveARVoiceSession",
    "ar_live_websocket",
    "ar_router",
    "chat_ar_artifact",
    "chat_historyverse_ar_mode_artifact",
    "historyverse_ar_mode_router",
    "legacy_ar_router",
    "start_ar_conversation",
    "start_historyverse_ar_mode_conversation",
]
