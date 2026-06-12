# API Analyst Report: voice\voice_loop.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import inspect`
- `import logging`
- `from typing import Any`
- `from core.voice.stt import SpeechToText`
- `from core.voice.tts import TextToSpeech`
- `from core.voice.wake_word import WakeWordDetector`

## Schemas & API Contracts (Classes)

### Class `VoiceLoop`
**Methods:**
- `def __init__(self, controller: Any, config: Any) -> None`
- `async def run(self) -> None`
- `async def _process_text(self, text: str) -> str`
- `async def ask_confirm(self, prompt: str) -> bool`
- `async def stop(self) -> None`

