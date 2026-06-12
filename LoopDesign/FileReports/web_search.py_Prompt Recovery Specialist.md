# File Report: web_search.py
**Role**: Prompt Recovery Specialist

## Dependencies
- re
- typing
- logging
- core.controller.request_rules
- __future__
- core.tools.web_tools

## Configuration Variables & Constants
- `synthesis_prompt`: (Too long, 223 chars. Extracted to Prompts if applicable)

## Schemas & API Contracts
### Function `handle_web_search`
**Args**: user_input, trace_id, memory, llm, model_router, profile
**Assumptions/Doc**: Perform a live web search, synthesize a natural language response, and fall back if needed.

### Function `_dispatch_llm_fallback`
**Args**: user_input, trace_id, memory, llm, model_router, profile
**Assumptions/Doc**: Clean fallback to raw LLM completion when search tool fails or is disabled.

### Function `_format_raw_fallback`
**Args**: raw_results
**Assumptions/Doc**: Parse and format raw search results nicely when LLM synthesis is not available.

## Prompts and LLM Directives
- Extracted `synthesis_prompt` to Prompts directory.
