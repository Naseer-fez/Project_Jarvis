"""Voice loop orchestrator for JarvisControllerV2."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.voice.stt import SpeechToText
from core.voice.tts import TextToSpeech
from core.voice.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceLoop:
    """wake_word -> STT -> controller.process -> TTS"""

    def __init__(self, controller: Any, config: Any) -> None:
        self.controller = controller
        self.config = config
        self.stt = SpeechToText(config)
        self.tts = TextToSpeech(config)
        self.wake = WakeWordDetector(config)
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("Voice loop started")
        print("Voice mode active. Say wake word or type input in fallback mode.")

        while self._running:
            wake_detected = await self.wake.wait_for_wake()
            if not wake_detected or not self._running:
                continue

            text = (await self.stt.transcribe()).strip()
            if not text:
                continue

            text_l = text.lower()
            if text_l in {"exit", "quit", "stop", "stop voice"}:
                self._running = False
                break

            print(f"You (voice): {text}")
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self.controller.process, text)
            print(f"Jarvis: {response}")
            await self.tts.speak(response)

        logger.info("Voice loop stopped")

    async def ask_confirm(self, prompt: str) -> bool:
        """Voice confirmation hook for high-risk actions."""
        await self.tts.speak(prompt)
        text = (await self.stt.transcribe()).strip().lower()
        return text in {"yes", "y", "yeah", "proceed", "allow", "confirm"}

    async def stop(self) -> None:
        self._running = False
        self.wake.stop()


__all__ = ["VoiceLoop"]
