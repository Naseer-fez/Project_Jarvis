# File Report: home_assistant.py
**Path**: `d:\AI\Jarvis\integrations\clients\home_assistant.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- os
- time
- typing.Any
- urllib.parse.quote
- integrations.base.BaseIntegration
- aiohttp
- aiohttp

## Classes and State Objects
### `HomeAssistantIntegration`
**Variables**: name, description
**Methods**: __init__, is_available, get_tools, execute, _base_url, _headers, _request, _read_response, _extract_error_message, _invalidate_entity_cache, _get_states, _extract_entity_ids, _build_target, _infer_domain, _contains_sensitive_domain, _normalize_service_data, _format_entity, _get_entity_state, _entity_service, _set_thermostat, _call_service, _list_entities, _format_service_result

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_entity_state",
                "description": "Get the current state and friendly name for a Home Assistant entity",
                "risk": "low",
                "args": {
                    "entity_id": {"type": "string", "description": "Entity ID like light.kitchen"},
                },
                "required_args": ["entity_id"],
            },
            {
                "name": "turn_on_entity",
                "description": "Turn on a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data like brightness_pct",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "turn_off_entity",
                "description": "Turn off a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "toggle_entity",
                "description": "Toggle a Home Assistant light, switch, fan, or similar entity",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional Home Assistant area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional Home Assistant device ID", "default": ""},
                    "domain": {"type": "string", "description": "Required when targeting an area or device", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra service data",
                        "default": {},
                    },
                },
                "required_args": [],
            },
            {
                "name": "set_thermostat",
                "description": "Set a climate entity target temperature",
                "risk": "confirm",
                "args": {
                    "entity_id": {"type": "string", "description": "Climate entity ID like climate.living_room"},
                    "temperature": {"type": "number", "description": "Target temperature"},
                    "hvac_mode": {
                        "type": "string",
                        "description": "Optional HVAC mode like heat, cool, or auto",
                        "default": "",
                    },
                },
                "required_args": ["entity_id", "temperature"],
            },
            {
                "name": "call_service",
                "description": "Call a specific Home Assistant service for a targeted entity, area, or device",
                "risk": "confirm",
                "args": {
                    "domain": {"type": "string", "description": "Service domain like light or media_player"},
                    "service": {"type": "string", "description": "Service name like turn_on or volume_set"},
                    "entity_id": {"type": "string", "description": "Optional single entity ID", "default": ""},
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of entity IDs",
                        "default": [],
                    },
                    "area_id": {"type": "string", "description": "Optional area ID", "default": ""},
                    "device_id": {"type": "string", "description": "Optional device ID", "default": ""},
                    "service_data": {
                        "type": "object",
                        "description": "Optional extra Home Assistant service fields",
                        "default": {},
                    },
                },
                "required_args": ["domain", "service"],
            },
            {
                "name": "list_entities",
                "description": "List Home Assistant entities, optionally filtered by domain",
                "risk": "low",
                "args": {
                    "domain": {"type": "string", "description": "Optional domain filter like light", "default": ""},
                    "include_attributes": {
                        "type": "boolean",
                        "description": "Include raw Home Assistant attributes in the response",
                        "default": False,
                    },
                },
                "required_args": [],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.