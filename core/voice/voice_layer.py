"""Thin wiring layer around VoiceLoop."""

from __future__ import annotations

import asyncio
from typing import Any

from core.voice.voice_loop import VoiceLoop


class VoiceLayer:
    def __init__(self, controller: Any, config: Any) -> None:
        self._loop = VoiceLoop(controller=controller, config=config)
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop.run(), name="jarvis_voice_loop")

    async def run(self) -> None:
        await self._loop.run()

    async def ask_confirm(self, prompt: str) -> bool:
        return await self._loop.ask_confirm(prompt)

    async def stop(self) -> None:
        await self._loop.stop()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


__all__ = ["VoiceLayer"]
