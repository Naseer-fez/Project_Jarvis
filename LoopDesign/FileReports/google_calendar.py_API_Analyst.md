# clients/google_calendar.py API Analyst Report

## Overview
Google Calendar v3 integration via async REST calls and OAuth2 token refresh.

## API Contracts & Methods
- `GoogleCalendarIntegration(BaseIntegration)`
  - Uses `aiohttp` to target `https://www.googleapis.com/calendar/v3`.

## Tools Exposed
- `create_event(summary, start, end, description, timezone, calendar_id="primary")` [Risk: `confirm`]
  - Dates must be ISO-8601. Converts to RFC3339 for Google APIs.
- `list_events(days_ahead=7, max_results=10, calendar_id="primary")` [Risk: `low`]
- `delete_event(event_id, calendar_id="primary")` [Risk: `confirm`]
- `find_free_slot(duration_minutes=60, days_ahead=7, calendar_id="primary")` [Risk: `low`]
  - Uses the `freeBusy` endpoint and calculates gaps between busy times locally.

## Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

## Assumptions & Constants
- Dates default to UTC.
- `find_free_slot` iterates busy periods sequentially and stops at the first large enough gap.

## Dependencies
- `aiohttp`
- `datetime`

## Prompts
- None.
