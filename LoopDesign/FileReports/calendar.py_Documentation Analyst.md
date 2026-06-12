# Documentation Report: clients/calendar.py

## Assumptions
- Backed by local `.ics` file at `memory/calendar.ics`
- Mutex lock `_calendar_lock` is used for thread-safe concurrent writes to the file.
- Events are appended manually by splitting on `END:VCALENDAR`.
- `add_event` assumes duration_minutes=60 if not specified.
- `list_events` uses `icalendar` and `dateutil` to parse the file and filter events occurring between now and `days_ahead`.

## Schema / API Contract
- Tool: `add_event(title: str, date: str (YYYY-MM-DD), time: str (HH:MM), duration_minutes: int)` -> `{"event", "date", "time"}`
- Tool: `list_events(days_ahead: int)` -> `{"events": [{"title", "datetime"}]}`

## Dependencies
- `icalendar`, `dateutil` (external)
- `asyncio`, `datetime`, `threading`, `pathlib` (stdlib)

## Configuration Variables
None.

## Prompts
None.
