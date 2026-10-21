"""Text-to-speech synthesis service."""

from __future__ import annotations

import logging
import os
import re

from agentverse.config.settings import Settings, get_settings
from agentverse.core.base import BaseTTSProvider
from agentverse.core.exceptions import VoiceError

logger = logging.getLogger(__name__)

__all__ = ["GTTSTextToSpeech"]


class GTTSTextToSpeech(BaseTTSProvider):
    """Text-to-speech engine using gTTS and language detection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def synthesize(self, text: str, output_path: str | None = None) -> str:
        """Synthesize text to audio and return file path."""
        cleaned = re.sub(r"[*#_`]", "", text).strip()
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
        from agentverse.utils.multilingual import detect_script

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
