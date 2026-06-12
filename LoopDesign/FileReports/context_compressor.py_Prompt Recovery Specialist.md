# File Report: context_compressor.py
**Role**: Prompt Recovery Specialist

## Dependencies
- hashlib
- datetime
- re
- math
- typing
- logging
- asyncio
- __future__

## Configuration Variables & Constants
- `TITLE_SYSTEM`: (Too long, 146 chars. Extracted to Prompts if applicable)
- `prompt`: (Too long, 250 chars. Extracted to Prompts if applicable)
- `prompt`: (Too long, 103 chars. Extracted to Prompts if applicable)
- `entry`: `f-string: {...}={...}`
- `entry`: `f-string: "{...}" -> "{...}"`

## Schemas & API Contracts
### Class `ContextCompressor`
**Assumptions/Doc**: Compress recalled memory into a compact block for LLM injection.
**Methods**: __init__, compress, _get_item_text, _summarize_context, _compress_preferences, _compress_episodes, _compress_conversations, _generate_focus_title, _clean, _truncate, _estimate_tokens, _deduplicate, explain

### Function `__init__`
**Args**: self, max_tokens, threshold, llm, enable_llm_title

### Function `compress`
**Args**: self, query, recall_results, include_scores
**Assumptions/Doc**: Build a compact text block from structured recall results with aging and deduplication.

### Function `_get_item_text`
**Args**: self, item

### Function `_summarize_context`
**Args**: self, top_memories

### Function `_compress_preferences`
**Args**: self, prefs, include_scores

### Function `_compress_episodes`
**Args**: self, episodes, include_scores

### Function `_compress_conversations`
**Args**: self, convos, include_scores

### Function `_generate_focus_title`
**Args**: self, query, lines
**Assumptions/Doc**: Generate a short semantic title using the quick-task LLM route.

### Function `_clean`
**Args**: text

### Function `_truncate`
**Args**: text, max_len

### Function `_estimate_tokens`
**Args**: text

### Function `_deduplicate`
**Args**: items, key

### Function `explain`
**Args**: self, query, recall_results
**Assumptions/Doc**: Return a human-readable explanation of what would be included.

## Prompts and LLM Directives
- Extracted `TITLE_SYSTEM` to Prompts directory.
- Extracted `prompt` to Prompts directory.
- Extracted `prompt` to Prompts directory.
- Extracted `entry` to Prompts directory.
- Extracted `entry` to Prompts directory.
