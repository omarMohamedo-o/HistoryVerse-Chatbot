"""Acoustic and Linguistic Speech Sanitization for Voice Services.

Provides text preprocessing for natural Speech-to-Text (STT) transcript ingestion
and Text-to-Speech (TTS) acoustic synthesis.
"""

from __future__ import annotations

import re

from agent.services.nlp.arabic import clean_arabic_text
from agent.services.nlp.colloquial import expand_egyptian_colloquial_query
from agent.services.security.pii import redact_egyptian_pii


def sanitize_for_speech(text: str) -> str:
    """Prepares text for natural speech synthesis.

    Strips unpronounceable metadata, square bracket citations, PII redaction tokens,
    markdown symbols, and raw URLs so acoustic voice output is fluent and immersive.
    """
    if not text:
        return ""

    # Remove square bracket citations and redaction markers: e.g. [المصدر: ...], [رقم_قومي_محجوب]
    cleaned = re.sub(r"\[(?:المصدر|مرجع|كتاب|بردي|سجل|Source|Ref|رقم|REDACTED)[^\]]*\]", "", text)
    # Convert markdown links [title](url) to title
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Remove raw URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Remove markdown formatting characters
    cleaned = re.sub(r"[*#_`>~]", "", cleaned)
    # Remove remaining bare brackets
    cleaned = re.sub(r"[\[\]{}]", "", cleaned)
    # Normalize multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_speech_transcript(raw_text: str, expand_dialect: bool = True) -> str:
    """Sanitizes raw Whisper STT transcript.

    Redacts spoken Egyptian PII, normalizes Arabic characters and numerals,
    and optionally expands Egyptian colloquial phrases into formal search concepts.
    """
    if not raw_text:
        return ""

    sanitized = redact_egyptian_pii(raw_text)
    normalized = clean_arabic_text(sanitized)
    if expand_dialect:
        return expand_egyptian_colloquial_query(normalized)
    return normalized
