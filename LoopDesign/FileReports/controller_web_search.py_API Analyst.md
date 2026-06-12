# API Analyst Report: controller\web_search.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `import re`
- `from typing import Any`
- `from core.controller.request_rules import is_explicit_web_search`
- `from core.controller.request_rules import should_force_web_search`

## Functions & Endpoints

### `handle_web_search`
`async def handle_web_search(user_input: str, trace_id: str, memory: Any, llm: Any, model_router: Any, profile: Any) -> str`
> Perform a live web search, synthesize a natural language response, and fall back if needed.

### `_dispatch_llm_fallback`
`async def _dispatch_llm_fallback(user_input: str, trace_id: str, memory: Any, llm: Any, model_router: Any, profile: Any) -> str`
> Clean fallback to raw LLM completion when search tool fails or is disabled.

### `_format_raw_fallback`
`def _format_raw_fallback(raw_results: str) -> str`
> Parse and format raw search results nicely when LLM synthesis is not available.
