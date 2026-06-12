# API Analyst Report: controller\intent_router.py

## Dependencies
- `from __future__ import annotations`
- `from dataclasses import dataclass`
- `from typing import Any`
- `from typing import Awaitable`
- `from typing import Callable`

## Schemas & API Contracts (Classes)

### Class `IntentRoute`
**Fields/Schema:**
  - `condition: Callable[[str, str, Any], bool]`
  - `handler: Callable[[str, str, Any], Awaitable[str | None]]`



### Class `IntentRouter`
**Methods:**
- `def __init__(self)`
- `def register(self, condition: Callable[[str, str, Any], bool], handler: Callable[[str, str, Any], Awaitable[str | None]]) -> None`
- `async def route(self, lowered: str, user_input: str, context: Any) -> str | None`

