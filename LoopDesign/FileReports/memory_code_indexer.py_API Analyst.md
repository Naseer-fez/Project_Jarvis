# API Analyst Report: memory\code_indexer.py

## Dependencies
- `import ast`
- `from typing import Any`

## Functions & Endpoints

### `_fallback_file_chunk`
`def _fallback_file_chunk(file_path: str, content: str, *, chunk_type: str, error: str | None=None) -> dict[str, Any]`
### `extract_code_chunks`
`def extract_code_chunks(file_path: str, content: str) -> list[dict[str, Any]]`
> Parse Python code and extract class/function chunks for semantic retrieval.
