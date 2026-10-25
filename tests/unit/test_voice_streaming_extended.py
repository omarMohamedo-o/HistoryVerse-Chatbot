"""Unit tests for voice streaming, language classification, and audio chunking edge cases."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.services.voice.streaming_stt import IncrementalSTT
from agent.services.voice.streaming_tts import SentenceChunker, StreamingTTS
from agent.services.voice.stt import WhisperSTT
from agent.services.voice.tts import GTTSTextToSpeech


def test_tts_language_detection_scripts():
    """Verify GTTSTextToSpeech detects Cyrillic, Greek, and CJK scripts."""
    tts = GTTSTextToSpeech()
    # Cyrillic -> ru
    assert tts._detect_language("Привет, это древняя история") == "ru"
    # Greek -> el
    assert tts._detect_language("Αρχαία ελληνική ιστορία και ναοί") == "el"
    # CJK -> zh-CN
    assert tts._detect_language("古埃及法老与金字塔的历史") == "zh-CN"


def test_whisper_stt_model_initialization(monkeypatch):
    """Verify WhisperSTT initializes and returns faster-whisper model."""
    mock_fw = MagicMock()
    mock_model_instance = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model_instance
    monkeypatch.setitem(sys.modules, "faster_whisper", mock_fw)

    stt = WhisperSTT()
    model = stt._ensure_model()
    assert model is mock_model_instance


@pytest.mark.asyncio
async def test_streaming_stt_edge_cases():
    """Verify IncrementalSTT handles empty chunks, half-second estimation, and batch fallback."""
    mock_base_stt = MagicMock()
    mock_base_stt.transcribe.return_value = "Batch transcript result"

    stt = IncrementalSTT(base_stt=mock_base_stt, sample_rate=16000)

    # 1. Fallback to batch transcribe
    assert stt.transcribe("/tmp/audio.wav") == "Batch transcript result"

    # 2. Empty chunk returns partial and False
    partial, is_final = await stt.feed_audio_chunk(b"")
    assert partial == ""
    assert is_final is False

    # 3. Feed 0.5s of audio to trigger partial transcript
    half_sec_bytes = b"\x00\x01" * 8000  # 16000 bytes = 0.5s at 16kHz 16-bit
    partial, is_final = await stt.feed_audio_chunk(half_sec_bytes)
    assert "[Processing" in partial

    # 4. Corrupt RMS energy bytes
    with patch("struct.unpack", side_effect=ValueError("Corrupt struct")):
        energy = stt._compute_rms_energy(b"\x00\x01")
        assert energy == 100.0


@pytest.mark.asyncio
async def test_streaming_tts_empty_feed_and_synthesis(monkeypatch):
    """Verify SentenceChunker empty token handling and StreamingTTS synthesis."""
    chunker = SentenceChunker()
    # Empty token feed
    assert chunker.feed("") == []

    # Mock gTTS for synthesis
    mock_gtts_mod = MagicMock()
    mock_tts_inst = MagicMock()
    mock_tts_inst.write_to_fp.side_effect = lambda fp: fp.write(b"AUDIO_DATA_BYTES")
    mock_gtts_mod.gTTS.return_value = mock_tts_inst
    monkeypatch.setitem(sys.modules, "gtts", mock_gtts_mod)

    streaming_tts = StreamingTTS()
    audio = await streaming_tts.synthesize_chunk("Hello world")
    assert audio == b"AUDIO_DATA_BYTES"
