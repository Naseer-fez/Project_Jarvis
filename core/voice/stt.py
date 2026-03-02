"""Speech-to-text for voice mode with faster-whisper fallback to keyboard input."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment]


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
            logger.info("STT backend: faster-whisper")
            return "faster-whisper"
        logger.warning("STT backend fallback: keyboard input")
        return "input"

    async def transcribe(self) -> str:
        if self._backend == "input":
            return await self._read_from_input()

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._record_and_transcribe)
        text = text.strip()
        if text:
            return text

        # Runtime fallback for model/audio failures.
        return await self._read_from_input()

    async def _read_from_input(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input("You (voice fallback): ").strip())

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed")
        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def _record_audio(self):
        if sd is None or np is None:
            return None

        frames = int(self.max_duration_s * self.sample_rate)
        recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        return recording.flatten()

    def _record_and_transcribe(self) -> str:
        try:
            self._ensure_model()
            audio = self._record_audio()
            if audio is None or len(audio) == 0:
                return ""

            segments, _ = self._model.transcribe(audio, language=self.language)
            text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT failed: %s", exc)
            return ""


STT = SpeechToText

__all__ = ["SpeechToText", "STT"]
