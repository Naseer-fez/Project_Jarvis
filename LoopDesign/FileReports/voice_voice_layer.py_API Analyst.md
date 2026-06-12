# API Analyst Report: voice\voice_layer.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `from typing import Any`
- `from core.voice.voice_loop import VoiceLoop`

## Schemas & API Contracts (Classes)

### Class `VoiceLayer`
**Methods:**
- `def __init__(self, controller: Any, config: Any) -> None`
- `async def start(self) -> None`
- `async def run(self) -> None`
- `async def ask_confirm(self, prompt: str) -> bool`
- `async def stop(self) -> None`

