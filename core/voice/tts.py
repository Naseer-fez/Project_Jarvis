"""Text-to-speech helpers with graceful fallback chain."""

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
except ImportError:  # optional dependency
    edge_tts = None  # type: ignore[assignment]

try:
    import pyttsx3
except ImportError:  # optional dependency
    pyttsx3 = None  # type: ignore[assignment]


class TextToSpeech:
    """TTS engine chain: edge-tts -> pyttsx3 -> print."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self.preferred = self._get("tts_engine", "edge-tts").lower()
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
        """Speak text with configured fallback order."""
        if not text:
            return

        if self.preferred == "edge-tts":
            if await self._speak_edge(text):
                return
            if await self._speak_pyttsx3(text):
                return
            print(f"Jarvis: {text}")
            return

        if self.preferred in {"pyttsx3", "offline"}:
            if await self._speak_pyttsx3(text):
                return
            if await self._speak_edge(text):
                return
            print(f"Jarvis: {text}")
            return

        # Unknown engine setting: still try both sensible options.
        if await self._speak_edge(text):
            return
        if await self._speak_pyttsx3(text):
            return
        print(f"Jarvis: {text}")

    async def _speak_edge(self, text: str) -> bool:
        if edge_tts is None:
            return False

        try:
            tmp_path = Path(tempfile.gettempdir()) / f"jarvis_tts_{abs(hash(text))}.mp3"
            communicator = edge_tts.Communicate(text=text, voice=self.voice_name)
            await communicator.save(str(tmp_path))

            if os.name == "nt":
                os.startfile(str(tmp_path))  # type: ignore[attr-defined]
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge-tts failed, falling back: %s", exc)
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
