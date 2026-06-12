# API Analyst Report: runtime\event_bus.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import threading`
- `import time`
- `import uuid`
- `from collections import deque`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from typing import Any`
- `from typing import Awaitable`
- `from typing import Callable`
- `from typing import Union`

## Schemas & API Contracts (Classes)

### Class `EventRecord`
> Replayable event envelope stored by the local event bus.

**Fields/Schema:**
  - `event_id: str`
  - `event_type: str`
  - `payload: Any`
  - `source: str`
  - `created_at: float`

**Methods:**
- `def to_dict(self) -> dict[str, Any]`


### Class `EventBus`
> Publish/Subscribe Event Bus allowing loose coupling between modules.

**Methods:**
- `def __init__(self, *, history_limit: int=500) -> None`
- `def _try_capture_loop(self) -> None`
- `def subscribe(self, event_type: str, callback: EventCallback, *, replay_history: bool=False) -> None`
  - *Register a callback for a specific event type.*
- `def unsubscribe(self, event_type: str, callback: EventCallback) -> None`
  - *Unregister a callback for a specific event type.*
- `def publish(self, event_type: str, data: Any, *, source: str='runtime') -> EventRecord`
  - *Publish an event to all registered subscribers.*
- `async def publish_async(self, event_type: str, data: Any, *, source: str='runtime') -> EventRecord`
  - *Asynchronously publish an event to all registered subscribers.*
- `def replay(self, event_type: str | None=None, *, limit: int | None=None) -> list[EventRecord]`
  - *Return recent events, optionally filtered by type.*
- `def clear_history(self) -> None`
- `def _record(self, event_type: str, payload: Any, *, source: str) -> EventRecord`
- `def _callbacks_for(self, event_key: str) -> list[EventCallback]`
- `def _dispatch_callback(self, callback: EventCallback, data: Any, event_key: str) -> None`

