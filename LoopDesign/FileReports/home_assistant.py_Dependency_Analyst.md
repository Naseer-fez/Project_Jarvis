# File Report: home_assistant.py
## Role: Dependency Analyst

### 1. Library Requirements
- `aiohttp` (Third-party)
- `os`, `time`, `typing`, `urllib.parse.quote` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- A reachable Home Assistant instance API.

### 3. Hidden Execution Links
- Maintains an in-memory `_entity_cache` valid for 60 seconds.
- Modifying entities (via service calls) aggressively invalidates the cache so subsequent reads see the updated state.
- Protects against domain inference interacting with high-risk operations: `_contains_sensitive_domain` verifies if `lock` or `alarm_control_panel` are being manipulated via generic helper functions and errors out, demanding explicit service calls.

### 4. Assumptions & API Contracts
- `turn_on_entity`, `turn_off_entity`, and `toggle_entity` act as "convenience helpers" predicting the domain (e.g. `light.kitchen` -> domain `light`).
- Automatically extracts the domain from `entity_id` string when targeting specific devices.
- Removes trailing slashes from the base URL configuration. If user provides URL ending with `/api`, strips it to form the base.

### 5. Configuration Variables
- `HOME_ASSISTANT_URL`
- `HOME_ASSISTANT_TOKEN`
- Internal: `_ENTITY_CACHE_TTL_SECONDS` (60), `_SENSITIVE_DOMAINS` (`{"lock", "alarm_control_panel"}`)

### 6. Prompts Found
- None.
