# Analysis Report for client.py

## Dependencies
- __future__.annotations
- asyncio
- logging
- re
- time
- pathlib.Path
- typing.Any
- core.config.defaults.OLLAMA_BASE_URL
- core.llm.ollama_client.OllamaClient
- core.llm.model_router.ModelRouter
- core.llm.defaults.DEFAULT_MODEL

## Schemas
- LLMClientV2

## API Contracts
- _strip_fences(text)
- _get_workspace_map(path, max_depth, max_files)
- LLMClientV2.__init__(self, hybrid_memory, model, profile, base_url, max_concurrent)
- LLMClientV2.set_router(self, router)
- LLMClientV2.set_telemetry(self, telemetry)
- LLMClientV2._record_telemetry(self, model, task_type, latency_ms, prompt, response, success)
- LLMClientV2.chat(self, messages, query_for_memory, profile_summary, workspace_path, trace_id, task_type)
- LLMClientV2._messages_to_prompt(messages)

## Configuration Variables
- JARVIS_SYSTEM
- _WORKSPACE_CACHE (typed)

## Assumptions & Notes
- Module Docstring: Async LLM client — single entry point for all Jarvis LLM calls.

Architecture:
    LLMClientV2.complete(prompt, task_type)
        → ModelRouter.pick_model(task_type)   → model name
        → OllamaClient.complete(prompt, model) → response (or raise)
        → CloudLLMClient.complete(prompt)      → fallback response

