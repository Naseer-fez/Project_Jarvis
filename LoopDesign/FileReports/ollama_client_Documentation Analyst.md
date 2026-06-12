# Analysis Report for ollama_client.py

## Dependencies
- __future__.annotations
- asyncio
- json
- logging
- re
- typing.Any
- urllib.request.urlopen
- aiohttp
- core.config.defaults.OLLAMA_BASE_URL
- core.llm.defaults.DEFAULT_MODEL

## Schemas
- OllamaTransientError
- OllamaClient

## API Contracts
- _normalize_base_url(base_url)
- _strip_think(text)
- extract_model_names(payload)
- list_models_sync(base_url, timeout_s)
- OllamaClient.__init__(self, base_url)

## Configuration Variables
- TIMEOUT_S

## Assumptions & Notes
- Module Docstring: Pure async HTTP client for local Ollama inference.

Talks to Ollama's /api/generate endpoint.  No cloud fallback,
no memory injection, no profile — just HTTP in, text out.

