"""Unit tests for streaming STT, sentence chunking, and streaming TTS."""

import struct
from unittest.mock import MagicMock

import pytest

from agent.config.settings import Settings
from agent.services.voice.streaming_stt import IncrementalSTT
from agent.services.voice.streaming_tts import SentenceChunker, StreamingTTS


def test_sentence_chunker_english():
    chunker = SentenceChunker(min_chunk_chars=10)
    tokens = ["Hello ", "world. ", "This ", "is ", "a ", "test sentence! ", "And ", "more."]
    sentences = []
    for token in tokens:
        sentences.extend(chunker.feed(token))
    remainder = chunker.flush()
    if remainder:
        sentences.append(remainder)

    assert len(sentences) >= 2
    assert "Hello world." in sentences[0]
    assert "This is a test sentence!" in sentences[1]


def test_sentence_chunker_arabic():
    chunker = SentenceChunker(min_chunk_chars=10)
    tokens = [
        "أهلاً ",
        "بكم في ",
        "متحف الإسكندرية. ",
        "هذا ",
        "تمثال ",
        "بطليموس الثالث، ",
        "وهو رائع.",
    ]
    sentences = []
    for token in tokens:
        sentences.extend(chunker.feed(token))
    remainder = chunker.flush()
    if remainder:
        sentences.append(remainder)

    assert len(sentences) >= 2
    assert "متحف الإسكندرية." in sentences[0]


@pytest.mark.asyncio
async def test_sentence_chunker_chunk_stream():
    chunker = SentenceChunker(min_chunk_chars=5)

    async def token_gen():
        yield "Welcome to "
        yield "the museum. "
        yield "Explore ancient "
        yield "Egypt!"

    results = []
    async for sentence in chunker.chunk_stream(token_gen()):
        results.append(sentence)

    assert len(results) == 2
    assert results[0] == "Welcome to the museum."
    assert results[1] == "Explore ancient Egypt!"


@pytest.mark.asyncio
async def test_streaming_tts_synthesize_chunk_and_stream():
    settings = Settings()
    tts = StreamingTTS(settings=settings)

    # Test synthesize_chunk
    audio_bytes = await tts.synthesize_chunk("This is a synthesized test clause.")
    assert audio_bytes is not None
    assert len(audio_bytes) > 0

    # Test synthesize_chunk empty
    assert await tts.synthesize_chunk("   ") is None

    # Test stream_speech
    async def token_gen():
        yield "The Great Sphinx is "
        yield "in Giza. It is majestic!"

    chunks = []
    async for chunk in tts.stream_speech(token_gen()):
        chunks.append(chunk)

    assert len(chunks) >= 1
    assert all(isinstance(c, bytes) for c in chunks)


@pytest.mark.asyncio
async def test_incremental_stt_energy_vad_and_finish(tmp_path):
    mock_base_stt = MagicMock()
    mock_base_stt.transcribe.return_value = "Rosetta Stone transcription"

    stt = IncrementalSTT(
        base_stt=mock_base_stt,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        silence_threshold_energy=50.0,
        silence_duration_seconds=0.1,
    )

    # Generate synthetic silent PCM (16000 samples/sec * 2 bytes = 32000 bytes/sec)
    # 0.05s of silence = 1600 bytes of zeros
    silence_chunk = b"\x00" * 1600
    _, is_final_1 = await stt.feed_audio_chunk(silence_chunk)
    assert not is_final_1

    # Generate synthetic audible PCM (sine wave / high amplitude)
    loud_samples = [(10000 * (1 if i % 2 == 0 else -1)) for i in range(800)]
    loud_chunk = struct.pack(f"<{len(loud_samples)}h", *loud_samples)

    _, is_final_2 = await stt.feed_audio_chunk(loud_chunk)
    assert not is_final_2

    # Feed enough silence to trigger VAD endpoint
    for _ in range(5):
        _, is_final_3 = await stt.feed_audio_chunk(silence_chunk)
        if is_final_3:
            break

    # Finalize transcription
    final_text = await stt.finish()
    assert final_text == "Rosetta Stone transcription"
    mock_base_stt.transcribe.assert_called_once()
