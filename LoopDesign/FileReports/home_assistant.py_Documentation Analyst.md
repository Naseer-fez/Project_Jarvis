# Documentation Report: clients/home_assistant.py

## Assumptions
- Interacts with Home Assistant REST API.
- Caches entity states (`/api/states`) for 60 seconds (`_ENTITY_CACHE_TTL_SECONDS`) to avoid spamming HA.
- Distinguishes "sensitive" domains (e.g. `lock`, `alarm_control_panel`) which cannot be modified via convenient toggle endpoints and must go through explicit confirm-gated services.
- Can extract target entity_id, area_id, device_id.

## Schema / API Contract
- Tools: `get_entity_state`, `turn_on_entity`, `turn_off_entity`, `toggle_entity`, `set_thermostat`, `call_service`, `list_entities`.

## Dependencies
- `aiohttp` (external)
- `os`, `time`, `urllib.parse` (stdlib)

## Configuration Variables
- `HOME_ASSISTANT_URL`
- `HOME_ASSISTANT_TOKEN`

## Prompts
None.
