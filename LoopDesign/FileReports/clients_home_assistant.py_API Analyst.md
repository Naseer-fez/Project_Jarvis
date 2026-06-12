# `home_assistant.py` - API Analyst Report

## Overview
Home Assistant integration via the REST API for reading entity states and controlling smart-home devices.

## Endpoints / Tools
1. `get_entity_state`
   - Description: Get the current state and friendly name for an entity.
   - Risk: low (read-only)
   - Arguments: `entity_id` (string, required).
2. `turn_on_entity`, `turn_off_entity`, `toggle_entity`
   - Description: Control a Home Assistant light, switch, fan, etc.
   - Risk: confirm (write)
   - Arguments: `entity_id` (string), `entity_ids` (array), `area_id` (string), `device_id` (string), `domain` (string), `service_data` (object).
3. `set_thermostat`
   - Description: Set a climate entity target temperature.
   - Risk: confirm (write)
   - Arguments: `entity_id` (string, required), `temperature` (number, required), `hvac_mode` (string).
4. `call_service`
   - Description: Call a specific Home Assistant service for an entity, area, or device.
   - Risk: confirm (write)
   - Arguments: `domain` (string, required), `service` (string, required), `entity_id` (string), `entity_ids` (array), `area_id` (string), `device_id` (string), `service_data` (object).
5. `list_entities`
   - Description: List Home Assistant entities, optionally filtered by domain.
   - Risk: low (read-only)
   - Arguments: `domain` (string), `include_attributes` (boolean, default False).

## External Contracts / Dependencies
- Requires `HOME_ASSISTANT_URL`, `HOME_ASSISTANT_TOKEN`.
- Strips trailing slashes and optionally trailing `/api` from `HOME_ASSISTANT_URL`.
- Uses `aiohttp` for HTTP requests to `/api/states`, `/api/states/{entity_id}`, `/api/services/{domain}/{service}`.

## Assumptions
- Implements a caching mechanism for `/api/states` results (`_entity_cache`) with a TTL of 60 seconds (`_ENTITY_CACHE_TTL_SECONDS`). This cache is invalidated on write actions (service calls).
- Refuses to run convenience helpers (`turn_on`, `turn_off`, `toggle`) for sensitive domains (`lock`, `alarm_control_panel`). They must be explicitly called via `call_service`.
- In `turn_on`/`off`/`toggle` helpers, if the target is an area or device, `domain` is required. Otherwise, it is inferred from `entity_id` (e.g., `light.kitchen` -> `light`).
