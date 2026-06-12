# API Analyst Report: controller\complexity_scorer.py

## Dependencies
- `from __future__ import annotations`
- `import re`
- `import logging`
- `from typing import Any`

## Configuration Variables
- `_REFLEX_KEYWORDS` = `{'weather', 'time', 'date', 'status', 'hello', 'hi', 'ping'}`
- `_DEEP_KEYWORDS` = `{'architecture', 'debug', 'refactor', 'complex', 'system design', 'explain how', 'why is this failing', 'optimize'}`
- `_AGENTIC_KEYWORDS` = `{'create', 'write', 'plan', 'workflow', 'automate', 'search', 'find', 'organize', 'download', 'fetch', 'open', 'launch', 'start', 'close', 'type', 'click', 'do', 'execute', 'run', 'make'}`
- `_CONDITIONAL_WORDS` = `{'if', 'when', 'unless', 'assuming', 'provided', 'suppose'}`
- `_TECHNICAL_TERMS` = `{'api', 'async', 'await', 'class', 'function', 'method', 'endpoint', 'database', 'schema', 'deploy', 'container', 'docker', 'kubernetes', 'pipeline', 'microservice', 'oauth', 'jwt', 'websocket', 'regex'}`

## Functions & Endpoints

### `_structural_signals`
`def _structural_signals(text: str) -> dict[str, Any]`
> Extract structural signals from the raw input.

### `classify_request`
`def classify_request(user_input: str) -> dict[str, Any]`
> Classify the complexity and type of request to determine routing.

Classes: Reflex, Chat, Agentic, Deep_Reasoning.

Returns a dict with routing metadata *and* enriched signals:
``class``, ``complexity``, ``route``, ``skip_planner``,
``estimated_tokens``, ``needs_reasoning``, ``needs_tools``,
``needs_vision``, ``context_weight``.
