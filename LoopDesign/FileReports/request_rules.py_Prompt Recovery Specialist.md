# File Report: request_rules.py
**Role**: Prompt Recovery Specialist

## Dependencies
- re
- __future__

## Configuration Variables & Constants

## Schemas & API Contracts
### Function `looks_like_desktop_control_request`
**Args**: lowered

### Function `is_explicit_web_search`
**Args**: lowered
**Assumptions/Doc**: Return True when the user unambiguously asks for a live web search.

### Function `should_force_web_search`
**Args**: lowered

### Function `is_active_window_request`
**Args**: lowered

### Function `is_preference_relevant`
**Args**: key, query
**Assumptions/Doc**: Determine if a retrieved preference key is relevant to the user query.

### Function `clean`
**Args**: s

## Prompts and LLM Directives
No explicit prompts found in module scope.
