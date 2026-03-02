"""Voice loop orchestration: wake -> transcribe -> process -> speak."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from core.voice.stt import SpeechToText
from core.voice.tts import TextToSpeech
from core.voice.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceLoop:
    def __init__(self, controller: Any, config: Any) -> None:
        self.controller = controller
        self.config = config

        self.stt = SpeechToText(config)
        self.tts = TextToSpeech(config)
        self.wake = WakeWordDetector(config)

        self._running = False

    async def run(self) -> None:
        self._running = True
        print("Voice mode active. Say wake word or type text in fallback mode. Say 'exit' to stop.")
        logger.info("Voice loop started")

        while self._running:
            triggered = await self.wake.wait_for_wake()
            if not triggered or not self._running:
                continue

            text = (await self.stt.transcribe()).strip()
            if not text:
                continue

            lowered = text.lower()
            if lowered in {"exit", "quit", "stop voice", "stop"}:
                self._running = False
                break

            response = await self._process_text(text)
            response = (response or "").strip() or "I do not have a response yet."

            print(f"You (voice): {text}")
            print(f"Jarvis: {response}")
            await self.tts.speak(response)

        logger.info("Voice loop stopped")

    async def _process_text(self, text: str) -> str:
        process_fn = getattr(self.controller, "process", None)
        if process_fn is None:
            return "Controller has no process() method."

        if inspect.iscoroutinefunction(process_fn):
            return str(await process_fn(text))

        loop = asyncio.get_running_loop()
        return str(await loop.run_in_executor(None, process_fn, text))

    async def ask_confirm(self, prompt: str) -> bool:
        await self.tts.speak(prompt)
        answer = (await self.stt.transcribe()).strip().lower()
        return answer in {"y", "yes", "yeah", "confirm", "allow", "proceed"}

    async def stop(self) -> None:
        self._running = False
        self.wake.stop()


__all__ = ["VoiceLoop"]
