# API Analyst Report: memory\sqlite_storage.py

## Dependencies
- `import json`
- `import logging`
- `from datetime import datetime`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `SQLiteStorage`
> Handles raw SQLite operations for memory (preferences, episodes, conversations, actions).

**Methods:**
- `def __init__(self, pool)`
- `async def init_schema(self) -> None`
- `async def store_preference(self, key: str, value: str) -> None`
- `async def store_episode(self, event: str, category: str='general') -> None`
- `async def store_episodes_batch(self, events: list[str], category: str='general') -> None`
- `async def store_conversation(self, user_input: str, assistant_response: str, session_id: str) -> None`
- `async def store_action(self, action: str, result: str, success: bool, metadata: dict | None) -> None`
- `async def recent_actions(self, limit: int=20) -> list[dict[str, Any]]`

