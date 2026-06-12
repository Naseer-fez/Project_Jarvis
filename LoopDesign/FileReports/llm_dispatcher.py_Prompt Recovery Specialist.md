# File Report: llm_dispatcher.py
**Role**: Prompt Recovery Specialist

## Dependencies
- typing
- logging
- core.controller.request_rules
- __future__

## Configuration Variables & Constants
- `profile_summary`: ``
- `task_type`: `reflex`
- `task_type`: `deep_reasoning`
- `task_type`: `planning`
- `task_type`: `chat`

## Schemas & API Contracts
### Class `LLMDispatcher`
**Assumptions/Doc**: Routes classified requests to the appropriate LLM model via the adaptive router.
**Methods**: __init__, dispatch

### Function `__init__`
**Args**: self, llm, model_router, memory, profile

### Function `dispatch`
**Args**: self, text, classification, session_id, trace_id

## Prompts and LLM Directives
No explicit prompts found in module scope.
