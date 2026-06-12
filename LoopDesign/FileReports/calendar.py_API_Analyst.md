# clients/calendar.py API Analyst Report

## Overview
A local calendar integration that operates entirely on an `.ics` file stored in memory (`memory/calendar.ics`).

## API Contracts & Methods
- `CalendarIntegration(BaseIntegration)`
  - `is_available()`: Checks for `icalendar` and `dateutil` packages.
  - `execute(tool_name, args)`: Routes to `_add_event` or `_list_events`.

## Tools Exposed
- `add_event`
  - **Risk:** `confirm`
  - **Args:** `title` (str), `date` (str: YYYY-MM-DD), `time` (str: HH:MM, default "09:00"), `duration_minutes` (int, default 60)
  - **Behavior:** Appends a `VEVENT` block to `memory/calendar.ics`.
- `list_events`
  - **Risk:** `low`
  - **Args:** `days_ahead` (int, default 7)
  - **Behavior:** Parses the `.ics` file, returns events occurring between now and `now + days_ahead`.

## Assumptions & Constants
- Calendar path is hardcoded to `memory/calendar.ics`.
- Writes direct strings to `.ics` instead of using `icalendar` for generation.
- Timezone defaults to `tzlocal()`.

## Dependencies
- `icalendar`
- `dateutil`
- `asyncio`, `threading`

## Prompts
- None.
