"""Speech-to-text for voice mode.

Provides `STT` class with the API expected by V2 acceptance tests:
  - STT._ready (bool attribute)
  - STT.is_ready (property)
  - STT.capture_and_transcribe() -> str | None
  - STT._is_speech(pcm_bytes, frame_length) -> bool
  - STT._vad (porcupine VAD or None)
  - STT._sample_rate (int)
  - TranscriptResult dataclass

Also provides `SpeechToText` (async variant) for the new async controller path.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore[import]
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import sounddevice as sd  # type: ignore[import]
except ImportError:
    sd = None  # type: ignore[assignment]

try:
    from faster_whisper import WhisperModel  # type: ignore[import]
except ImportError:
    WhisperModel = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranscriptResult:
    """Structured output from speech recognition."""
    text: str
    audio_hash: str = ""
    duration_s: float = 0.0
    language: str = "en"
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# STT class — synchronous + attribute-based API (V2 acceptance tests)
# ─────────────────────────────────────────────────────────────────────────────

_ENERGY_THRESHOLD = 500  # RMS amplitude above which audio is considered speech


class STT:
    """
    Synchronous STT wrapper with graceful degradation.

    Attributes exposed for tests:
      _ready         — True once the backend model is loaded
      _vad           — VAD object or None when using energy-based detection
      _sample_rate   — audio sample rate in Hz
    """

    _VAD_ENERGY_THRESHOLD = _ENERGY_THRESHOLD

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._ready: bool = False
        self._vad = None
        self._sample_rate: int = 16_000
        self._model = None

        if config is not None:
            self._init(config)

    def _init(self, config: Any) -> None:
        """Attempt to load the whisper model. Sets _ready on success."""
        try:
            srate_raw = config.get("voice", "audio_sample_rate", fallback="16000")
            self._sample_rate = int(srate_raw)
        except Exception:  # noqa: BLE001
            self._sample_rate = 16_000

        if WhisperModel is not None:
            try:
                model_name = config.get("voice", "stt_model", fallback="base.en")
                compute_type = config.get("voice", "stt_compute_type", fallback="int8")
                self._model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
                self._ready = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Whisper model load failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── VAD ───────────────────────────────────────────────────────────────

    def _is_speech(self, pcm_bytes: bytes, frame_length: int) -> bool:
        """
        Return True if the PCM frame contains speech.

        When self._vad is None (no external VAD), uses simple energy-based
        detection: compute RMS of 16-bit signed samples.
        """
        if self._vad is not None:
            try:
                return bool(self._vad.is_speech(pcm_bytes, self._sample_rate))
            except Exception:  # noqa: BLE001
                pass

        # Energy-based VAD fallback
        try:
            num_samples = len(pcm_bytes) // 2
            if num_samples == 0:
                return False
            samples = struct.unpack(f"{num_samples}h", pcm_bytes[: num_samples * 2])
            rms = (sum(s * s for s in samples) / num_samples) ** 0.5
            return rms > self._VAD_ENERGY_THRESHOLD
        except Exception:  # noqa: BLE001
            return False

    # ── Capture ───────────────────────────────────────────────────────────

    def capture_and_transcribe(self) -> Optional[str]:
        """
        Record audio and return the transcribed text (or None if not ready).
        Falls back to None when no audio device is available.
        """
        if not self._ready:
            return None

        if sd is None or np is None:
            return None

        try:
            duration = 5
            recording = sd.rec(
                int(duration * self._sample_rate),
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = recording.flatten()

            if self._model is None:
                return None

            segments, _ = self._model.transcribe(audio, language="en")
            text = " ".join(s.text.strip() for s in segments if s.text).strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT capture_and_transcribe failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# SpeechToText — async variant for the async controller path
# ─────────────────────────────────────────────────────────────────────────────

class SpeechToText:
    """Async STT wrapper with graceful fallback behavior."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.model_name = self._get("stt_model", "base")
        self.language = self._get("stt_language", "en")
        self.sample_rate = int(self._get("audio_sample_rate", "16000"))
        self.max_duration_s = float(self._get("stt_max_duration_s", "8"))

        self._backend = self._choose_backend()
        self._model = None

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    def _choose_backend(self) -> str:
        if WhisperModel is not None and np is not None and sd is not None:
            return "faster-whisper"
        return "input"

    async def transcribe(self) -> str:
        if self._backend == "input":
            return await self._read_from_input()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._record_and_transcribe)
        return text.strip() or await self._read_from_input()

    async def _read_from_input(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input("You (voice fallback): ").strip())

    def _record_and_transcribe(self) -> str:
        try:
            if WhisperModel is None or np is None or sd is None:
                return ""
            if self._model is None:
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            frames = int(self.max_duration_s * self.sample_rate)
            recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            segments, _ = self._model.transcribe(recording.flatten(), language=self.language)
            return " ".join(s.text.strip() for s in segments if s.text).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT failed: %s", exc)
            return ""


__all__ = ["STT", "SpeechToText", "TranscriptResult"]
