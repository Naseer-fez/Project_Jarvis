# File Report: ollama_client.py
**Role**: Prompt Recovery Specialist

## Dependencies
- re
- urllib.request
- typing
- json
- logging
- asyncio
- __future__
- core.llm.defaults
- aiohttp
- core.config.defaults

## Configuration Variables & Constants

## Schemas & API Contracts
### Class `OllamaTransientError`
**Assumptions/Doc**: Raised when Ollama returns a transient HTTP error status.
**Methods**: 

### Class `OllamaClient`
**Assumptions/Doc**: Lightweight async client for a local Ollama instance.

Usage::

    client = OllamaClient()
    reply = await client.complete("say hello", model="mistral:7b")
**Methods**: __init__, complete, list_models, is_running

### Function `_normalize_base_url`
**Args**: base_url

### Function `_strip_think`
**Args**: text
**Assumptions/Doc**: Remove <think>…</think> blocks emitted by DeepSeek R1.

### Function `extract_model_names`
**Args**: payload

### Function `list_models_sync`
**Args**: base_url, timeout_s

### Function `__init__`
**Args**: self, base_url

### Function `complete`
**Args**: self, prompt, system, temperature
**Assumptions/Doc**: Send a prompt to Ollama and return the response text.

Retries up to 3 times on transient connection errors.
Raises on timeout, connection refused, or empty response.

### Function `list_models`
**Args**: self
**Assumptions/Doc**: Return currently available Ollama model tags.

### Function `is_running`
**Args**: self
**Assumptions/Doc**: Quick health check — GET the Ollama root endpoint.

## Prompts and LLM Directives
No explicit prompts found in module scope.
