# File Report: cloud_client.py
**Role**: Prompt Recovery Specialist

## Dependencies
- logging
- aiohttp
- os
- __future__

## Configuration Variables & Constants
- `url`: `f-string: https://generativelanguage.googleapis.com/v1beta/models/{...}:generateContent?key={...}`

## Schemas & API Contracts
### Class `CloudLLMClient`
**Assumptions/Doc**: Best-effort cloud fallback across a small provider chain, with Tier-aware routing.
**Methods**: __init__, complete, _call, _call_groq, _call_openai, _call_anthropic, _call_gemini, _extract_openai_usage

### Function `__init__`
**Args**: self

### Function `complete`
**Args**: self, prompt, system, temperature, tier

### Function `_call`
**Args**: self, provider, prompt, system, temperature, model

### Function `_call_groq`
**Args**: self, prompt, system, temperature, model

### Function `_call_openai`
**Args**: self, prompt, system, temperature, model

### Function `_call_anthropic`
**Args**: self, prompt, system, temperature, model

### Function `_call_gemini`
**Args**: self, prompt, system, temperature, model

### Function `_extract_openai_usage`
**Args**: data
**Assumptions/Doc**: Extract usage tokens from OpenAI-compatible API responses (OpenAI, Groq).

## Prompts and LLM Directives
- Extracted `url` to Prompts directory.
