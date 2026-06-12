# File Report: google_calendar.py
**Path**: `d:\AI\Jarvis\integrations\clients\google_calendar.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- os
- datetime.datetime
- typing.Any
- integrations.base.BaseIntegration
- aiohttp
- aiohttp
- aiohttp
- aiohttp
- aiohttp
- aiohttp

## Classes and State Objects
### `GoogleCalendarIntegration`
**Variables**: name, description
**Methods**: is_available, get_tools, execute, _refresh_access_token, _to_rfc3339, _create_event, _list_events, _delete_event, _find_free_slot

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_event",
                "description": "Create a new event in Google Calendar",
                "risk": "confirm",
                "args": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start": {
                        "type": "string",
                        "description": "Start datetime in ISO-8601 (e.g. 2026-03-15T10:00:00)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime in ISO-8601",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                        "default": "",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone (e.g. Asia/Kolkata)",
                        "default": "UTC",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                        "default": "primary",
                    },
                },
                "required_args": ["summary", "start", "end"],
            },
            {
                "name": "list_events",
                "description": "List upcoming Google Calendar events",
                "risk": "low",
                "args": {
                    "days_ahead": {"type": "integer", "description": "Look ahead N days", "default": 7},
                    "max_results": {"type": "integer", "description": "Max events to return", "default": 10},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": [],
            },
            {
                "name": "delete_event",
                "description": "Delete a Google Calendar event by its event ID",
                "risk": "confirm",
                "args": {
                    "event_id": {"type": "string", "description": "Google Calendar event ID"},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": ["event_id"],
            },
            {
                "name": "find_free_slot",
                "description": "Find the next free time slot of a given duration",
                "risk": "low",
                "args": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Required free slot duration in minutes",
                        "default": 60,
                    },
                    "days_ahead": {"type": "integer", "default": 7},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": [],
            },
        ]

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.