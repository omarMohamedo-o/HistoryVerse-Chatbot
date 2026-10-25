"""Speech-to-text transcription service using Faster-Whisper."""

from __future__ import annotations

import logging
import os

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseSTTProvider
from agent.core.exceptions import VoiceError

logger = logging.getLogger(__name__)

__all__ = ["WhisperSTT"]


class WhisperSTT(BaseSTTProvider):
    """Whisper-based audio transcriber with lazy model loading."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model size=%s on device=%s",
                self._settings.whisper_model_size,
                self._settings.whisper_device,
            )
            self._model = WhisperModel(
                self._settings.whisper_model_size,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
                cpu_threads=min(4, os.cpu_count() or 1),
            )
            return self._model
        except Exception as exc:
            logger.error("Failed to initialize Faster-Whisper: %s", exc)
            raise VoiceError(f"Whisper initialization failure: {exc}") from exc

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file into text."""
        if not os.path.exists(audio_path):
            raise VoiceError(f"Audio file not found: {audio_path}")

        model = self._ensure_model()
        try:
            segments, _ = model.transcribe(audio_path)
            raw_transcript = "".join(segment.text for segment in segments).strip()

            # Apply global security and NLP sanitization
            from agent.services.voice.sanitizer import sanitize_speech_transcript

            transcript = sanitize_speech_transcript(raw_transcript, expand_dialect=False)
            logger.info("Transcribed and sanitized audio (%d chars)", len(transcript))
            return transcript
        except Exception as exc:
            logger.error("Speech transcription failed: %s", exc)
            raise VoiceError(f"Transcription error: {exc}") from exc
