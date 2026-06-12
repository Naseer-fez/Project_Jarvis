# File Report: client.py
**Role**: Prompt Recovery Specialist

## Dependencies
- core.llm.model_router
- concurrent.futures
- time
- core.llm.cloud_client
- re
- typing
- logging
- asyncio
- __future__
- core.llm.ollama_client
- core.llm.defaults
- os
- pathlib
- core.config.defaults

## Configuration Variables & Constants
- `JARVIS_SYSTEM`: (Too long, 125 chars. Extracted to Prompts if applicable)
- `response`: ``
- `profile_injection`: ``
- `style_instruction`: ``
- `cloud_model`: `f-string: cloud_tier{...}`
- `context`: ``

## Schemas & API Contracts
### Class `LLMClientV2`
**Assumptions/Doc**: Public interface — all LLM calls in Jarvis enter here.

Wiring (in order):
    1. ``ModelRouter.pick_model(task_type)`` → model name
    2. ``OllamaClient.complete(prompt, model=…)`` → try local first
    3. ``CloudLLMClient.complete(prompt)`` → fallback if Ollama fails
**Methods**: __init__, set_router, set_telemetry, complete, _record_telemetry, chat_async, chat, _build_system, _messages_to_prompt

### Function `_strip_fences`
**Args**: text

### Function `_get_workspace_map`
**Args**: path, max_depth, max_files
**Assumptions/Doc**: Build a compact directory view to ground model responses in local files.

### Function `_walk`
**Args**: current, depth

### Function `__init__`
**Args**: self, hybrid_memory, model, profile, base_url, max_concurrent

### Function `set_router`
**Args**: self, router

### Function `set_telemetry`
**Args**: self, telemetry
**Assumptions/Doc**: Connect execution telemetry for recording LLM call metrics.

### Function `complete`
**Args**: self, prompt, system, temperature, task_type, keep_think, classification
**Assumptions/Doc**: Text completion: ModelRouter → OllamaClient → CloudLLMClient fallback.

Steps:
    1. Ask ModelRouter for the best model name for this task_type.
    2. Call OllamaClient.complete() with that model.
    3. If response quality is poor, auto-escalate once.
    4. If Ollama raises ANY exception, fall back to CloudLLMClient
       with the correct tier.
    5. Record telemetry for every call.

### Function `_record_telemetry`
**Args**: self, model, task_type, latency_ms, prompt, response, success
**Assumptions/Doc**: Record call metrics to telemetry if available.

### Function `chat_async`
**Args**: self, messages, query_for_memory, profile_summary, workspace_path, trace_id, task_type
**Assumptions/Doc**: Async version — use this inside any async context (agent loop, controller).

### Function `chat`
**Args**: self, messages, query_for_memory, profile_summary, workspace_path, trace_id, task_type
**Assumptions/Doc**: Sync bridge — ONLY call from truly synchronous, non-async contexts.

### Function `_build_system`
**Args**: self, query, profile, workspace_path

### Function `_messages_to_prompt`
**Args**: messages

## Prompts and LLM Directives
- Extracted `JARVIS_SYSTEM` to Prompts directory.
- Extracted `style_instruction` to Prompts directory.
- Extracted `cloud_model` to Prompts directory.
