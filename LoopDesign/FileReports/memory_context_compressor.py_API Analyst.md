# API Analyst Report: memory\context_compressor.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import re`
- `from typing import Any`

## Configuration Variables
- `DEFAULT_MAX_TOKENS` = `400`
- `DEFAULT_THRESHOLD` = `0.3`
- `MAX_PREF_ITEMS` = `6`
- `MAX_EPISODE_ITEMS` = `3`
- `MAX_CONVO_ITEMS` = `2`
- `MAX_VALUE_LEN` = `60`
- `MAX_EPISODE_LEN` = `100`
- `MAX_CONVO_LEN` = `120`
- `TITLE_TIMEOUT_S` = `4.0`
- `TITLE_SYSTEM` = `'You create short memory context titles for a local AI assistant. Return only a 3-7 word title. No quotes, no bullets, no punctuation-heavy output.'`

## Schemas & API Contracts (Classes)

### Class `ContextCompressor`
> Compress recalled memory into a compact block for LLM injection.

**Methods:**
- `def __init__(self, max_tokens: int=DEFAULT_MAX_TOKENS, threshold: float=DEFAULT_THRESHOLD, llm: Any | None=None, enable_llm_title: bool=False) -> None`
- `async def compress(self, query: str, recall_results: dict, include_scores: bool=False) -> str`
  - *Build a compact text block from structured recall results with aging and deduplication.*
- `def _get_item_text(self, item: dict) -> str`
- `async def _summarize_context(self, top_memories: list[dict]) -> str`
- `def _compress_preferences(self, prefs: list[dict], include_scores: bool) -> tuple[list[str], int]`
- `def _compress_episodes(self, episodes: list[dict], include_scores: bool) -> tuple[list[str], int]`
- `def _compress_conversations(self, convos: list[dict], include_scores: bool) -> tuple[list[str], int]`
- `async def _generate_focus_title(self, query: str, lines: list[str]) -> str`
  - *Generate a short semantic title using the quick-task LLM route.*
- @staticmethod
- `def _clean(text: str) -> str`
- @staticmethod
- `def _truncate(text: str, max_len: int) -> str`
- @staticmethod
- `def _estimate_tokens(text: str) -> int`
- @staticmethod
- `def _deduplicate(items: list[dict], key: str) -> list[dict]`
- `def explain(self, query: str, recall_results: dict) -> str`
  - *Return a human-readable explanation of what would be included.*

