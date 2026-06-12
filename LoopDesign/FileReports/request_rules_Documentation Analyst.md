# Analysis Report for request_rules.py

## Dependencies
- __future__.annotations

## Schemas
None

## API Contracts
- looks_like_desktop_control_request(lowered)
- is_explicit_web_search(lowered)
- should_force_web_search(lowered)
- is_active_window_request(lowered)
- is_preference_relevant(key, query)

## Configuration Variables
- DESKTOP_CONTROL_KEYWORDS
- AGENTIC_KEYWORDS
- WEB_SEARCH_EXPLICIT_PHRASES
- LIVE_WEB_HINTS
- LIVE_WEB_REQUEST_MARKERS
- ACTIVE_WINDOW_PHRASES

## Assumptions & Notes
- Module Docstring: Reusable request-classification rules for controller routing.

