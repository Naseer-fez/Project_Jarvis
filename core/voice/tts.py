"""Text-to-speech with fallback chain: edge-tts -> pyttsx3 -> print."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import edge_tts
except ImportError:
    edge_tts = None  # type: ignore[assignment]

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # type: ignore[assignment]


class TextToSpeech:
    """Async TTS wrapper that never crashes when optional deps are missing."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.preferred_engine = self._get("tts_engine", "edge-tts").strip().lower()
        self.voice_name = self._get("tts_voice", "en-US-GuyNeural")
        self._pyttsx3_engine = None

        if pyttsx3 is not None:
            try:
                self._pyttsx3_engine = pyttsx3.init()
            except Exception as exc:  # noqa: BLE001
                logger.warning("pyttsx3 init failed: %s", exc)
                self._pyttsx3_engine = None

    def _get(self, key: str, default: str) -> str:
        try:
            return str(self._config.get("voice", key, fallback=default))
        except Exception:  # noqa: BLE001
            return default

    async def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        chain = self._engine_chain()
        for engine in chain:
            if engine == "edge" and await self._speak_edge(text):
                return
            if engine == "pyttsx3" and await self._speak_pyttsx3(text):
                return

        print(f"Jarvis: {text}")

    def _engine_chain(self) -> list[str]:
        if self.preferred_engine in {"edge", "edge-tts"}:
            return ["edge", "pyttsx3", "print"]
        if self.preferred_engine in {"pyttsx3", "offline"}:
            return ["pyttsx3", "edge", "print"]
        return ["edge", "pyttsx3", "print"]

    async def _speak_edge(self, text: str) -> bool:
        if edge_tts is None:
            return False

        try:
            temp_file = Path(tempfile.gettempdir()) / f"jarvis_tts_{abs(hash(text))}.mp3"
            communicator = edge_tts.Communicate(text=text, voice=self.voice_name)
            await communicator.save(str(temp_file))

            if os.name == "nt":
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, os.startfile, str(temp_file))  # type: ignore[attr-defined]
                return True

            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge-tts failed: %s", exc)
            return False

    async def _speak_pyttsx3(self, text: str) -> bool:
        if self._pyttsx3_engine is None:
            return False

        def _run() -> bool:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("pyttsx3 speak failed: %s", exc)
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run)


TTS = TextToSpeech

__all__ = ["TextToSpeech", "TTS"]
