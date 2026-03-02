"""Speech-to-text helpers for Jarvis voice mode."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:  # optional dependency
    np = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:  # optional dependency
    sd = None  # type: ignore[assignment]

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:  # optional dependency
    FasterWhisperModel = None  # type: ignore[assignment]

try:
    import whisper
except ImportError:  # optional dependency
    whisper = None  # type: ignore[assignment]


class SpeechToText:
    """Offline-first STT with graceful CLI fallback."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.model_name = self._get("stt_model", "base")
        self.language = self._get("stt_language", "en")
        self.sample_rate = int(self._get("audio_sample_rate", "16000"))
        self.record_seconds = float(self._get("stt_max_duration_s", "8"))

        self._backend = self._select_backend()
        self._model = None

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    def _select_backend(self) -> str:
        if FasterWhisperModel is not None and sd is not None and np is not None:
            logger.info("STT backend selected: faster-whisper")
            return "faster-whisper"
        if whisper is not None and sd is not None and np is not None:
            logger.info("STT backend selected: whisper")
            return "whisper"

        logger.warning(
            "STT model/audio dependencies missing; falling back to CLI text input"
        )
        return "cli"

    async def transcribe(self) -> str:
        """Capture one utterance and return text."""
        if self._backend == "cli":
            return await self._read_from_cli()

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._record_and_transcribe)
        if text:
            return text

        # If model/audio path fails at runtime, fall back without crashing voice loop.
        return await self._read_from_cli()

    async def _read_from_cli(self) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: input("You (voice fallback): ").strip())

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if self._backend == "faster-whisper":
            self._model = FasterWhisperModel(self.model_name, device="cpu", compute_type="int8")
            return

        if self._backend == "whisper":
            self._model = whisper.load_model(self.model_name)
            return

    def _record_audio(self):
        if sd is None or np is None:
            return None

        frames = int(self.record_seconds * self.sample_rate)
        recording = sd.rec(frames, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        return recording.flatten()

    def _record_and_transcribe(self) -> str:
        try:
            self._ensure_model()
            audio = self._record_audio()
            if audio is None:
                return ""

            if self._backend == "faster-whisper":
                segments, _ = self._model.transcribe(audio, language=self.language)
                return " ".join(segment.text.strip() for segment in segments).strip()

            if self._backend == "whisper":
                result = self._model.transcribe(audio, language=self.language, fp16=False)
                return str(result.get("text", "")).strip()

            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("STT transcription failed: %s", exc)
            return ""


STT = SpeechToText

__all__ = ["SpeechToText", "STT"]
