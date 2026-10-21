"""Real-time streaming and incremental Speech-to-Text provider."""

from __future__ import annotations

import io
import logging
import wave

from agent.config.settings import Settings, get_settings
from agent.core.base import BaseStreamingSTTProvider
from agent.services.voice.stt import WhisperSTT

logger = logging.getLogger(__name__)

__all__ = ["IncrementalSTT"]


class IncrementalSTT(BaseStreamingSTTProvider):
    """Incremental STT buffer with voice activity and silence boundary detection."""

    def __init__(
        self,
        base_stt: WhisperSTT | None = None,
        settings: Settings | None = None,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        silence_threshold_energy: float = 120.0,
        silence_duration_seconds: float = 0.8,
    ) -> None:
        self._settings = settings or get_settings()
        self._stt = base_stt or WhisperSTT(self._settings)
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.silence_threshold_energy = silence_threshold_energy
        self.silence_duration_seconds = silence_duration_seconds

        self._buffer = bytearray()
        self._partial_transcript = ""
        self._consecutive_silent_chunks = 0
        self._bytes_per_second = sample_rate * sample_width * channels

    def transcribe(self, audio_path: str) -> str:
        """Fallback to batch transcribe."""
        return self._stt.transcribe(audio_path)

    async def feed_audio_chunk(self, chunk: bytes) -> tuple[str, bool]:
        """Accumulate an incoming raw PCM audio chunk and detect voice boundary.

        Returns:
            Tuple of (current_transcript, is_final).
        """
        if not chunk:
            return self._partial_transcript, False

        self._buffer.extend(chunk)

        # Basic energy-based VAD calculation on 16-bit PCM
        energy = self._compute_rms_energy(chunk)
        chunk_duration = (
            len(chunk) / float(self._bytes_per_second) if self._bytes_per_second else 0.0
        )

        if energy < self.silence_threshold_energy:
            self._consecutive_silent_chunks += 1
        else:
            self._consecutive_silent_chunks = 0

        total_silence_time = self._consecutive_silent_chunks * chunk_duration
        is_final = (
            len(self._buffer) >= self._bytes_per_second
            and total_silence_time >= self.silence_duration_seconds
        )

        # Incremental transcript estimation for non-empty buffer
        if len(self._buffer) >= self._bytes_per_second * 0.5:
            self._partial_transcript = f"[Processing {len(self._buffer)} bytes...]"

        return self._partial_transcript, is_final

    async def finish(self) -> str:
        """Finalize the current stream, convert buffer to WAV, and run transcription."""
        if not self._buffer:
            return ""

        wav_bytes = self._encode_wav(bytes(self._buffer))
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            try:
                transcript = self._stt.transcribe(tmp_path)
            finally:
                import os

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            self._partial_transcript = transcript
            return transcript
        except Exception as exc:
            logger.warning("Incremental STT transcription failed: %s", exc)
            return self._partial_transcript or ""
        finally:
            await self.reset()

    async def reset(self) -> None:
        """Reset the internal audio buffer and state."""
        self._buffer.clear()
        self._partial_transcript = ""
        self._consecutive_silent_chunks = 0

    def _compute_rms_energy(self, chunk: bytes) -> float:
        """Calculate Root Mean Square (RMS) energy for 16-bit linear PCM audio."""
        if len(chunk) < 2:
            return 0.0
        import struct

        count = len(chunk) // 2
        format_str = f"<{count}h"
        try:
            shorts = struct.unpack(format_str, chunk[: count * 2])
            sum_squares = sum(s * s for s in shorts)
            return (sum_squares / float(count)) ** 0.5
        except Exception:
            return 100.0

    def _encode_wav(self, pcm_data: bytes) -> bytes:
        """Pack raw PCM bytes into standard WAV container."""
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)
        return out.getvalue()
