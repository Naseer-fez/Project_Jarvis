# API Analyst Report: memory\code_indexer_service.py

## Dependencies
- `import asyncio`
- `import hashlib`
- `import logging`
- `from datetime import datetime`
- `from pathlib import Path`
- `from typing import Callable`

## Schemas & API Contracts (Classes)

### Class `CodeIndexerService`
> Handles codebase indexing and chunk extraction for hybrid memory.

**Methods:**
- `def __init__(self, db_pool, semantic_memory, store_episode_cb)`
- `async def index_codebase(self, root_path: str, is_hybrid: bool, init_schema_cb: Callable) -> dict[str, int]`
- `async def _index_chunks_batch(self, batch: list[dict], stats: dict, is_hybrid: bool, init_schema_cb: Callable) -> None`

