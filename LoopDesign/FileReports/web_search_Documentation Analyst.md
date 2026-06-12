# Analysis Report for web_search.py

## Dependencies
- __future__.annotations
- logging
- re
- typing.Any
- core.controller.request_rules.is_explicit_web_search
- core.controller.request_rules.should_force_web_search

## Schemas
None

## API Contracts
- _format_raw_fallback(raw_results)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Web search fast-path controller logic for Jarvis.
Handles explicit web searches directly, bypassing the full planner,
and synthesizes natural language responses.

