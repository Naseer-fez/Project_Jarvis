# API Analyst Report: llm\client.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import logging`
- `import re`
- `import time`
- `from pathlib import Path`
- `from typing import Any`
- `from core.config.defaults import OLLAMA_BASE_URL`
- `from core.llm.ollama_client import OllamaClient`
- `from core.llm.model_router import ModelRouter`
- `from core.llm.defaults import DEFAULT_MODEL`

## Configuration Variables
- `JARVIS_SYSTEM` = `"You are Jarvis, a local personal AI assistant.\nYou are concise, technical, and truthful.\nYou run on the user's local machine."`

## Prompts Extracted

- `style_instruction` -> Saved to `Prompts/client_style_instruction.txt`

## Schemas & API Contracts (Classes)

### Class `LLMClientV2`
> Public interface — all LLM calls in Jarvis enter here.

Wiring (in order):
    1. ``ModelRouter.pick_model(task_type)`` → model name
    2. ``OllamaClient.complete(prompt, model=…)`` → try local first
    3. ``CloudLLMClient.complete(prompt)`` → fallback if Ollama fails

**Methods:**
- `def __init__(self, hybrid_memory: Any=None, model: str=DEFAULT_MODEL, profile: Any=None, base_url: str=OLLAMA_BASE_URL, max_concurrent: int=4) -> None`
- `def set_router(self, router: ModelRouter) -> None`
- `def set_telemetry(self, telemetry: Any) -> None`
  - *Connect execution telemetry for recording LLM call metrics.*
- `async def complete(self, prompt: str, system: str='', temperature: float=0.1, task_type: str='chat', keep_think: bool=False, classification: dict[str, Any] | None=None) -> str`
  - *Text completion: ModelRouter → OllamaClient → CloudLLMClient fallback.*
- `def _record_telemetry(self, model: str, task_type: str, latency_ms: float, prompt: str, response: str, success: bool) -> None`
  - *Record call metrics to telemetry if available.*
- `async def chat_async(self, messages: list[dict[str, Any]], query_for_memory: str='', profile_summary: str='', workspace_path: str='', trace_id: str | None=None, task_type: str='chat') -> str`
  - *Async version — use this inside any async context (agent loop, controller).*
- `def chat(self, messages: list[dict[str, Any]], query_for_memory: str='', profile_summary: str='', workspace_path: str='', trace_id: str | None=None, task_type: str='chat') -> str`
  - *Sync bridge — ONLY call from truly synchronous, non-async contexts.*
- `async def _build_system(self, query: str='', profile: str='', workspace_path: str='') -> str`
- @staticmethod
- `def _messages_to_prompt(messages: list[dict[str, Any]]) -> str`


## Functions & Endpoints

### `_strip_fences`
`def _strip_fences(text: str) -> str`
### `_get_workspace_map`
`def _get_workspace_map(path: str, max_depth: int=3, max_files: int=50) -> str`
> Build a compact directory view to ground model responses in local files.
