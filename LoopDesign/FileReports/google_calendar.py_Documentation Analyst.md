# Documentation Report: clients/google_calendar.py

## Assumptions
- Uses Google Calendar v3 REST API directly via `aiohttp`.
- Refreshes tokens on-demand via Google OAuth token endpoint.
- Dates are passed natively as ISO-8601 strings, parsed to verify and injected with `timeZone` field in Google request payload.
- `find_free_slot` uses `freeBusy` endpoint to locate gaps between events over `days_ahead`.

## Schema / API Contract
- Tools: `create_event`, `list_events`, `delete_event`, `find_free_slot`.
- `create_event` accepts `summary`, `start`, `end`, `description`, `timezone`, `calendar_id`.

## Dependencies
- `aiohttp` (external)
- `os`, `datetime` (stdlib)

## Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

## Prompts
None.
