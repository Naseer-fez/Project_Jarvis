# File Report: google_calendar.py
## Role: Dependency Analyst

### 1. Library Requirements
- `aiohttp` (Third-party)
- `os`, `datetime`, `typing` (Standard Library)
- `integrations.base` (Local)

### 2. Service Dependencies
- Google OAuth API (`https://oauth2.googleapis.com/token`)
- Google Calendar API v3 (`https://www.googleapis.com/calendar/v3`)

### 3. Hidden Execution Links
- Refreshes the Google OAuth token on every call, identical to the Gmail integration.
- Parses ISO-8601 strings and handles naive vs offset-aware datetime before submitting to Google API via `_to_rfc3339()`.
- Calculates free slots using the `/freeBusy` endpoint and evaluating intervals locally.

### 4. Assumptions & API Contracts
- Uses `primary` calendar by default, overridable by `calendar_id`.
- Expects RFC3339 datetime strings, and will fallback between forms like `%Y-%m-%dT%H:%M:%S%z` and naive `%Y-%m-%dT%H:%M:%S` while passing the `timeZone` field (default UTC).
- In `_find_free_slot`, gaps are calculated strictly by checking adjacent busy slots against the current time (`datetime.now(tz=timezone.utc)`).

### 5. Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

### 6. Prompts Found
- None.
