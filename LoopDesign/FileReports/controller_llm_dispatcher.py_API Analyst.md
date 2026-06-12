# API Analyst Report: controller\llm_dispatcher.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `LLMDispatcher`
> Routes classified requests to the appropriate LLM model via the adaptive router.

**Methods:**
- `def __init__(self, llm, model_router, memory, profile)`
- `async def dispatch(self, text: str, classification: dict[str, Any], session_id: str, trace_id: str) -> str`

