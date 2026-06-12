# API Analyst Report: memory\retriever.py

## Dependencies
- `import logging`
- `import re`
- `from typing import Any`
- `from typing import Callable`

## Schemas & API Contracts (Classes)

### Class `MemoryRetriever`
> Handles hybrid search over semantic memory and SQLite storage.

**Methods:**
- `def __init__(self, db_pool, semantic_memory)`
- @staticmethod
- `def query_tokens(query: str) -> list[str]`
- @staticmethod
- `def score_text(text: str, tokens: list[str]) -> float`
- `async def recall_preferences(self, query: str, top_k: int, is_hybrid: bool, init_schema_cb: Callable) -> list[dict[str, Any]]`
- `async def _recall_sqlite_episodes(self, query: str, top_k: int, init_schema_cb: Callable) -> list[dict[str, Any]]`
- `async def _recall_sqlite_conversations(self, query: str, top_k: int, init_schema_cb: Callable) -> list[dict[str, Any]]`
- `async def recall_all(self, query: str, top_k: int, is_hybrid: bool, init_schema_cb: Callable) -> dict[str, list[dict[str, Any]]]`

