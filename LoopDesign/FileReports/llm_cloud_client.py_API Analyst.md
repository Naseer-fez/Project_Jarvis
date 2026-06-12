# API Analyst Report: llm\cloud_client.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `import os`

## Configuration Variables
- `PROVIDERS` = `['gemini', 'groq', 'openai', 'anthropic']`
- `MODELS` = `{'gemini': {1: 'gemini-2.0-flash-lite', 2: 'gemini-2.5-flash', 3: 'gemini-2.5-pro'}, 'groq': {1: 'llama-3.1-8b-instant', 2: 'llama-3.3-70b-versatile', 3: 'deepseek-r1-distill-llama-70b'}, 'openai': {1: 'gpt-4o-mini', 2: 'gpt-4o', 3: 'o3-mini'}, 'anthropic': {1: 'claude-3-haiku-20240307', 2: 'claude-3-5-sonnet-20241022', 3: 'claude-sonnet-4-20250514'}}`
- `PROVIDER_ORDER` = `{1: ['groq', 'gemini', 'openai', 'anthropic'], 2: ['groq', 'gemini', 'openai', 'anthropic'], 3: ['anthropic', 'openai', 'gemini', 'groq']}`

## Schemas & API Contracts (Classes)

### Class `CloudLLMClient`
> Best-effort cloud fallback across a small provider chain, with Tier-aware routing.

**Methods:**
- `def __init__(self) -> None`
- `async def complete(self, prompt: str, system: str='', temperature: float=0.1, tier: int=2) -> str`
- `async def _call(self, provider: str, prompt: str, system: str, temperature: float, model: str) -> tuple[str, dict[str, int]]`
- `async def _call_groq(self, prompt: str, system: str, temperature: float, model: str) -> tuple[str, dict[str, int]]`
- `async def _call_openai(self, prompt: str, system: str, temperature: float, model: str) -> tuple[str, dict[str, int]]`
- `async def _call_anthropic(self, prompt: str, system: str, temperature: float, model: str) -> tuple[str, dict[str, int]]`
- `async def _call_gemini(self, prompt: str, system: str, temperature: float, model: str) -> tuple[str, dict[str, int]]`
- @staticmethod
- `def _extract_openai_usage(data: dict) -> dict[str, int]`
  - *Extract usage tokens from OpenAI-compatible API responses (OpenAI, Groq).*

