# File Report: calendar.py
## Role: Dependency Analyst

### 1. Library Requirements
- `asyncio` (Standard Library)
- `datetime`, `threading`, `pathlib`, `typing` (Standard Library)
- `importlib.util` (Standard Library)
- `icalendar` (Third-party)
- `dateutil.tz` (Third-party from `python-dateutil`)
- `integrations.base` (Local): imports `BaseIntegration`

### 2. Service Dependencies
- None external. Acts on the local file system.

### 3. Hidden Execution Links
- Relies on `memory/calendar.ics` file acting as the local DB.
- `_add_event` writes directly to the `.ics` file using text manipulation if the parser isn't invoked.
- `_list_events` uses `icalendar.Calendar.from_ical` to parse the same file.
- Thread-safe writes via `threading.Lock()` (`_calendar_lock`).

### 4. Assumptions & API Contracts
- Assumes `icalendar` and `dateutil` are installed. If not, `is_available()` returns False.
- Assumes `memory/calendar.ics` path is valid/accessible.
- Events added without a timezone are treated as naive and parsed against the local timezone via `dateutil.tz.tzlocal()`.
- API uses standard risk conventions: `add_event` is `confirm`, `list_events` is `low`.

### 5. Configuration Variables
- No required env vars.
- Hardcoded path: `CALENDAR_PATH = Path("memory/calendar.ics")`.

### 6. Prompts Found
- None.
