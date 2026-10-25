"""Text-to-speech synthesis service."""

from __future__ import annotations

import logging
import os
import re

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseTTSProvider
from agent.core.exceptions import VoiceError
from agent.services.voice.sanitizer import sanitize_for_speech

logger = logging.getLogger(__name__)

__all__ = ["GTTSTextToSpeech", "sanitize_for_speech"]


class GTTSTextToSpeech(BaseTTSProvider):
    """Text-to-speech engine using gTTS and language detection."""

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            from agent.config.settings import get_settings
            self._settings = get_settings()
        else:
            self._settings = settings

    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Synthesize text to audio and return file path."""
        cleaned = sanitize_for_speech(text)
        if not cleaned:
            raise VoiceError("Cannot synthesize empty text.")

        out_dir = self._settings.outputs_dir
        os.makedirs(out_dir, exist_ok=True)
        target = (
            output_path if output_path else os.path.join(out_dir, f"tts_{os.urandom(4).hex()}.mp3")
        )

        lang = self._detect_language(cleaned)

        try:
            from gtts import gTTS

            tts = gTTS(text=cleaned, lang=lang, slow=False, tld=self._settings.tts_tld)
            tts.save(target)
            logger.info("Saved audio to %s (lang=%s)", target, lang)
            return target
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            raise VoiceError(f"TTS error: {exc}") from exc

    def _detect_language(self, text: str) -> str:
        from agent.utils.multilingual import detect_script

        script = detect_script(text)
        if script == "arabic":
            return "ar"
        if script == "cyrillic":
            return "ru"
        if script == "greek":
            return "el"
        if script == "cjk":
            return "zh-CN"

        try:
            import langid

            code, _ = langid.classify(text)
            return code
        except Exception:
            return self._settings.default_language
