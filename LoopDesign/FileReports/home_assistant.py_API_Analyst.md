# clients/home_assistant.py API Analyst Report

## Overview
Integration for Home Assistant, supporting reading entities and calling services via REST API.

## API Contracts & Methods
- `HomeAssistantIntegration(BaseIntegration)`
  - `_request(...)`: Handles HTTP calls using `aiohttp`.
  - `_get_states()`: Caches `/api/states` payload for 60 seconds.

## Tools Exposed
- `get_entity_state(entity_id)` [Risk: `low`]
- `turn_on_entity(...)` [Risk: `confirm`]
- `turn_off_entity(...)` [Risk: `confirm`]
- `toggle_entity(...)` [Risk: `confirm`]
  - All support generic payload shapes: `entity_id`, `entity_ids`, `area_id`, `device_id`, `domain`, `service_data`.
- `set_thermostat(entity_id, temperature, hvac_mode)` [Risk: `confirm`]
- `call_service(domain, service, ...)` [Risk: `confirm`]
- `list_entities(domain, include_attributes)` [Risk: `low`]

## Assumptions & Constants
- `_ENTITY_CACHE_TTL_SECONDS = 60`
- `_SENSITIVE_DOMAINS = {"lock", "alarm_control_panel"}`: Protected from being targeted by generic helpers like `turn_on_entity`.
- Normalizes errors to flat string messages.
- Removes trailing `/api` from URL if provided.

## Configuration Variables
- `HOME_ASSISTANT_URL`
- `HOME_ASSISTANT_TOKEN`

## Dependencies
- `aiohttp`
- `urllib.parse.quote`

## Prompts
- None.
