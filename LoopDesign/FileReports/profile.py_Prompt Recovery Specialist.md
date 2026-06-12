# File Report: profile.py
**Role**: Prompt Recovery Specialist

## Dependencies
- datetime
- pathlib
- threading
- json
- logging
- asyncio
- __future__
- os

## Configuration Variables & Constants

## Schemas & API Contracts
### Class `UserProfileEngine`
**Methods**: __init__, _fresh_defaults, _load, save, update_from_conversation, apply_delta, get_system_prompt_injection, get_communication_style, interaction_count

### Function `__init__`
**Args**: self

### Function `_fresh_defaults`
**Args**: self

### Function `_load`
**Args**: self

### Function `save`
**Args**: self
**Assumptions/Doc**: Atomic write to avoid corruption on interruption.

### Function `update_from_conversation`
**Args**: self, user_text, jarvis_response

### Function `apply_delta`
**Args**: self, delta, min_confidence
**Assumptions/Doc**: Apply synthesis delta and return list of updated fields.

### Function `get_system_prompt_injection`
**Args**: self
**Assumptions/Doc**: Compact profile context injected into the LLM system prompt.

### Function `get_communication_style`
**Args**: self

### Function `interaction_count`
**Args**: self

### Function `_write`
**Args**: 

## Prompts and LLM Directives
No explicit prompts found in module scope.
