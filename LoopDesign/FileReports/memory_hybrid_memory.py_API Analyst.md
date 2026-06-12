# API Analyst Report: memory\hybrid_memory.py

## Dependencies
- `from __future__ import annotations`
- `import contextlib`
- `import asyncio`
- `from pathlib import Path`
- `from typing import Any`
- `import logging`
- `from core.memory.semantic_memory import SemanticMemory`
- `from core.memory.sqlite_pool import SQLitePool`
- `from core.memory.sqlite_storage import SQLiteStorage`
- `from core.memory.retriever import MemoryRetriever`
- `from core.memory.code_indexer_service import CodeIndexerService`

## Schemas & API Contracts (Classes)

### Class `HybridMemory`
**Methods:**
- `def __init__(self, config_or_db_path: Any='memory/memory.db', chroma_path: str='data/chroma', model_name: str='all-MiniLM-L6-v2', *, db_path: str | None=None)`
- `async def initialize(self, index_path: str='') -> dict[str, Any]`
- `def _track_background_task(self, task: asyncio.Task[Any]) -> None`
- `def set_llm(self, llm: Any | None, *, enable_context_titles: bool=True) -> None`
- `async def _ensure_db_initialized(self) -> None`
- `async def _init_sqlite(self) -> None`
- `async def store_preference(self, key: str, value: str) -> bool`
- `async def store_episode(self, event: str, category: str='general') -> bool`
- `async def store_episodes_batch(self, events: list[str], category: str='general') -> bool`
- `async def store_conversation(self, user_input: str, assistant_response: str, session_id: str='default') -> bool`
- `async def recall_preferences(self, query: str, top_k: int=5) -> list[dict[str, Any]]`
- `async def recall_all(self, query: str, top_k: int=5) -> dict[str, list[dict[str, Any]]]`
- @staticmethod
- `def _query_tokens(query: str) -> list[str]`
- @staticmethod
- `def _score_text(text: str, tokens: list[str]) -> float`
- `async def _recall_sqlite_episodes(self, query: str, top_k: int=5) -> list[dict[str, Any]]`
- `async def _recall_sqlite_conversations(self, query: str, top_k: int=5) -> list[dict[str, Any]]`
- `async def build_context_block(self, query: str, n_results: int=5) -> str`
- `async def recall(self, query: str, top_k: int=5) -> str`
- `async def store_code_file(self, file_path: str, content: str) -> int`
- `async def index_codebase(self, root_path: str) -> dict[str, int]`
- `def stats(self) -> dict[str, Any]`
- `async def store_fact(self, key: str, value: str, source: str='user', **_kwargs) -> None`
- `async def get_fact(self, key: str)`
- `async def list_facts(self, limit: int=50) -> list`
- `async def count(self) -> int`
- `async def store_action(self, action: str, result: str='', success: bool=True, metadata: dict | None=None) -> None`
- `async def store_failure(self, action: str, error: str='', metadata: dict | None=None) -> None`
- `async def recent_actions(self, limit: int=20) -> list[dict[str, Any]]`
- `async def set_preference(self, key: str, value: str, category: str='general', **_kwargs) -> None`
- `async def get_preferences(self, category: str='') -> dict[str, str]`
- `async def close(self) -> None`


## Functions & Endpoints

### `_looks_like_config`
`def _looks_like_config(value: Any) -> bool`