# API Analyst Report: automation\rag_ingester.py

## Dependencies
- `from pathlib import Path`
- `from typing import Any`
- `import logging`

## Schemas & API Contracts (Classes)

### Class `RagIngester`
**Methods:**
- `def __init__(self, chunk_size_chars: int, chunk_overlap_chars: int, memory: Any, stats: Any)`
- `async def store_rag_text(self, *, source: str, path: Path, text: str) -> int`
- `def chunk_text(self, text: str) -> list[str]`

