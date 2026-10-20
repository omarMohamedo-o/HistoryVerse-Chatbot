"""Unit tests for Whisper STT and gTTS voice services."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from historyverse.config.settings import Settings
from historyverse.core.exceptions import VoiceError
from historyverse.services.voice.stt import WhisperSTT
from historyverse.services.voice.tts import GTTSTextToSpeech


def test_stt_missing_file_raises():
    stt = WhisperSTT()
    with pytest.raises(VoiceError, match="Audio file not found"):
        stt.transcribe("/tmp/nonexistent_audio_file.wav")


def test_stt_transcribe_mock(tmp_path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"RIFFdummywavdata")

    stt = WhisperSTT()

    mock_segment_1 = MagicMock()
    mock_segment_1.text = "Welcome to the "
    mock_segment_2 = MagicMock()
    mock_segment_2.text = "Alexandria library museum."

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment_1, mock_segment_2], None)

    stt._model = mock_model
    transcript = stt.transcribe(str(audio_file))
    assert transcript == "Welcome to the Alexandria library museum."


def test_tts_empty_text_raises():
    tts = GTTSTextToSpeech()
    with pytest.raises(VoiceError, match="Cannot synthesize empty text"):
        tts.synthesize("   ***   ")


def test_tts_synthesize_mock(tmp_path):
    tts = GTTSTextToSpeech(Settings())
    target_path = str(tmp_path / "output.mp3")

    mock_gtts_module = MagicMock()
    mock_gtts_instance = MagicMock()
    mock_gtts_module.gTTS.return_value = mock_gtts_instance

    mock_langid_module = MagicMock()
    mock_langid_module.classify.return_value = ("en", 0.99)

    with patch.dict(sys.modules, {"gtts": mock_gtts_module, "langid": mock_langid_module}):
        result = tts.synthesize("Welcome to the Valley of the Kings.", output_path=target_path)
        assert result == target_path
        mock_gtts_instance.save.assert_called_once_with(target_path)


def test_tts_detect_language():
    tts = GTTSTextToSpeech()
    mock_langid_module = MagicMock()
    mock_langid_module.classify.return_value = ("ar", 0.99)

    with patch.dict(sys.modules, {"langid": mock_langid_module}):
        lang = tts._detect_language("مرحبا بكم في مكتبة الاسكندرية")
        assert lang == "ar"
