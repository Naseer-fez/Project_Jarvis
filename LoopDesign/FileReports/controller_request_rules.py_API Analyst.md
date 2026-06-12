# API Analyst Report: controller\request_rules.py

## Dependencies
- `from __future__ import annotations`

## Configuration Variables
- `DESKTOP_CONTROL_KEYWORDS` = `('mouse', 'cursor', 'desktop', 'screen', 'keyboard', 'hotkey', 'click', 'scroll', 'drag', 'clipboard')`
- `AGENTIC_KEYWORDS` = `('search', 'look up', 'find', 'check', 'scrape', 'get', 'download', 'fetch', 'read', 'analyze', 'create', 'make', 'write', 'run', 'execute', 'automate', 'browse', 'internet', 'online', 'web', 'website', 'latest', 'current', 'today', 'live', 'news', 'price', 'weather', 'stats', 'score', 'runs', 'toss', 'ipl', 'match', 'mouse', 'cursor', 'desktop', 'screen', 'keyboard', 'hotkey', 'click', 'scroll', 'drag', 'clipboard')`
- `WEB_SEARCH_EXPLICIT_PHRASES` = `('search the web', 'search the internet', 'search online', 'browse the web', 'browse the internet', 'browse online', 'look it up online', 'look up online', 'google it', 'google for', 'find online', 'find on the internet', 'find on the web', 'search for', 'search web for', 'web search', 'internet search')`
- `LIVE_WEB_HINTS` = `('internet', 'online', 'web', 'website', 'latest', 'current', 'today', 'live', 'news', 'price', 'weather', 'score', 'stats', 'runs', 'toss', 'ipl', 'match')`
- `LIVE_WEB_REQUEST_MARKERS` = `('search', 'browse', 'find', 'check', 'look up', 'google', 'get', 'give me', 'tell me', 'update', 'use internet', 'what is', 'who is', 'when is')`
- `ACTIVE_WINDOW_PHRASES` = `('which app is active', 'what app is active', 'what window is active', 'which window is active', 'tell me the active window', 'get the active window')`

## Functions & Endpoints

### `looks_like_desktop_control_request`
`def looks_like_desktop_control_request(lowered: str) -> bool`
### `is_explicit_web_search`
`def is_explicit_web_search(lowered: str) -> bool`
> Return True when the user unambiguously asks for a live web search.

### `should_force_web_search`
`def should_force_web_search(lowered: str) -> bool`
### `is_active_window_request`
`def is_active_window_request(lowered: str) -> bool`
### `is_preference_relevant`
`def is_preference_relevant(key: str, query: str) -> bool`
> Determine if a retrieved preference key is relevant to the user query.
