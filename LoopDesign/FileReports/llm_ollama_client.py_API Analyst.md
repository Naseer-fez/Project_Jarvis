# API Analyst Report: llm\ollama_client.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import json`
- `import logging`
- `import re`
- `from typing import Any`
- `from urllib.request import urlopen`
- `import aiohttp`
- `from core.config.defaults import OLLAMA_BASE_URL`
- `from core.llm.defaults import DEFAULT_MODEL`

## Configuration Variables
- `TIMEOUT_S` = `120`

## Schemas & API Contracts (Classes)

### Class `OllamaTransientError(aiohttp.ClientError)`
> Raised when Ollama returns a transient HTTP error status.



### Class `OllamaClient`
> Lightweight async client for a local Ollama instance.

Usage::

    client = OllamaClient()
    reply = await client.complete("say hello", model="mistral:7b")

**Methods:**
- `def __init__(self, base_url: str=OLLAMA_BASE_URL) -> None`
- `async def complete(self, prompt: str, system: str='', temperature: float=0.1, *, model: str=DEFAULT_MODEL, keep_think: bool=False) -> str`
  - *Send a prompt to Ollama and return the response text.*
- `async def list_models(self) -> list[str]`
  - *Return currently available Ollama model tags.*
- `async def is_running(self) -> bool`
  - *Quick health check — GET the Ollama root endpoint.*


## Functions & Endpoints

### `_normalize_base_url`
`def _normalize_base_url(base_url: str) -> str`
### `_strip_think`
`def _strip_think(text: str) -> str`
> Remove <think>…</think> blocks emitted by DeepSeek R1.

### `extract_model_names`
`def extract_model_names(payload: dict[str, Any] | None) -> list[str]`
### `list_models_sync`
`def list_models_sync(base_url: str=OLLAMA_BASE_URL, timeout_s: float=3.0) -> list[str]`