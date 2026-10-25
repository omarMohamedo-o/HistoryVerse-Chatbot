"""Speech-to-text and Text-to-speech services."""

from __future__ import annotations

from agent.services.voice.sanitizer import sanitize_for_speech, sanitize_speech_transcript

__all__ = [
    "GTTSTextToSpeech",
    "IncrementalSTT",
    "SentenceChunker",
    "StreamingTTS",
    "WhisperSTT",
    "sanitize_for_speech",
    "sanitize_speech_transcript",
]


def __getattr__(name: str):
    if name == "IncrementalSTT":
        from agent.services.voice.streaming_stt import IncrementalSTT
        return IncrementalSTT
    if name in ("SentenceChunker", "StreamingTTS"):
        from agent.services.voice.streaming_tts import SentenceChunker, StreamingTTS
        return SentenceChunker if name == "SentenceChunker" else StreamingTTS
    if name == "WhisperSTT":
        from agent.services.voice.stt import WhisperSTT
        return WhisperSTT
    if name == "GTTSTextToSpeech":
        from agent.services.voice.tts import GTTSTextToSpeech
        return GTTSTextToSpeech
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
