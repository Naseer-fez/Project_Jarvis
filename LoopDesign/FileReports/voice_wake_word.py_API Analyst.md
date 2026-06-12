# API Analyst Report: voice\wake_word.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import os`
- `import threading`
- `from typing import Any`
- `from typing import Callable`
- `from typing import Optional`

## Schemas & API Contracts (Classes)

### Class `WakeWordDetector`
> Detects a wake word and fires callbacks; falls back to continuous mode
when porcupine is not installed.

Signature matches V2 acceptance tests:
  WakeWordDetector(config, loop, on_wake, on_cancel)

**Methods:**
- `def __init__(self, config: Any, loop: Optional[asyncio.AbstractEventLoop]=None, on_wake: Optional[Callable[[], None]]=None, on_cancel: Optional[Callable[[], None]]=None) -> None`
- `def _get(self, key: str, default: str) -> str`
- `def _fire_wake(self) -> None`
  - *Fire the on_wake callback, scheduling it on the event loop if provided.*
- `def _fire_cancel(self) -> None`
  - *Fire the on_cancel callback.*
- `async def wait_for_wake(self) -> bool`
  - *Return True when ready for STT capture.*
- `def _wait_blocking(self) -> bool`
- `def stop(self) -> None`
  - *Signal detection loop to halt.*

