# `google_calendar.py` - API Analyst Report

## Overview
Google Calendar integration via Google Calendar API v3 using fully async `aiohttp` requests.

## Endpoints / Tools
1. `create_event`
   - Description: Create a new event in Google Calendar.
   - Risk: confirm (write)
   - Arguments: `summary` (string, required), `start` (string, ISO-8601, required), `end` (string, ISO-8601, required), `description` (string), `timezone` (string, IANA, default "UTC"), `calendar_id` (string, default "primary").
2. `list_events`
   - Description: List upcoming events.
   - Risk: low (read-only)
   - Arguments: `days_ahead` (integer, default 7), `max_results` (integer, default 10), `calendar_id` (string, default "primary").
3. `delete_event`
   - Description: Delete a Google Calendar event.
   - Risk: confirm (write)
   - Arguments: `event_id` (string, required), `calendar_id` (string, default "primary").
4. `find_free_slot`
   - Description: Find the next free time slot using the `freeBusy` API.
   - Risk: low (read-only)
   - Arguments: `duration_minutes` (integer, default 60), `days_ahead` (integer, default 7), `calendar_id` (string, default "primary").

## External Contracts / Dependencies
- Requires `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`.
- OAuth token refresh endpoint: `https://oauth2.googleapis.com/token`.
- Base API URL: `https://www.googleapis.com/calendar/v3`.
- Uses `aiohttp` library.

## Assumptions
- Uses Google's `freeBusy` API to dynamically search for schedule gaps. Iterates over busy slots and returns the first gap `>= duration_minutes`.
- The `_to_rfc3339` method assumes datetimes are either `%Y-%m-%dT%H:%M:%S%z` or `%Y-%m-%dT%H:%M:%S` format, allowing naive datetimes to rely on the explicitly passed `timeZone` field.
- Maximum results for list limits at 50.
