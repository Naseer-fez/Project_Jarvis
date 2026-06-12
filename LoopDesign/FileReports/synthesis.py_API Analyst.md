# API Analyst Report: synthesis.py

## Dependencies
- `from __future__ import annotations`
- `import inspect`
- `import json`
- `import re`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `ProfileSynthesizer`
**Methods:**
- `def __init__(self, llm: Any) -> None`
- `def should_run(self, profile: Any) -> bool`
- `async def synthesize(self, conversations: list[str], profile: Any) -> dict[str, Any]`


## Functions & Endpoints

### `_strip_wrappers`
`def _strip_wrappers(text: str) -> str`