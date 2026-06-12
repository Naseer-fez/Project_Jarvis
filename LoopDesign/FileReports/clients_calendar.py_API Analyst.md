# `calendar.py` - API Analyst Report

## Overview
A local calendar integration using simple `.ics` (iCalendar) files to store and retrieve events.

## Endpoints / Tools
1. `add_event`
   - Description: Add an event to calendar.
   - Risk: confirm (write)
   - Arguments: `title` (string, required), `date` (string: YYYY-MM-DD, required), `time` (string: HH:MM, default 09:00), `duration_minutes` (integer, default 60).
2. `list_events`
   - Description: List upcoming events.
   - Risk: low (read-only)
   - Arguments: `days_ahead` (integer, default 7).

## External Contracts / Dependencies
- Relies on `icalendar` and `dateutil` Python packages for parsing `.ics` content.
- Reads and writes state to `memory/calendar.ics`.
- Emits events with a standard iCalendar schema.

## Assumptions
- Creates `memory/calendar.ics` if missing.
- Locks calendar file reading/writing behind a `threading.Lock()` (`_calendar_lock`) to prevent race conditions during concurrent modifications.
- Assumes local timezone using `dateutil.tz.tzlocal()`. Date-only events are assumed to be generated effectively in the local timezone and normalized to `datetime` objects.
- Uses `asyncio.get_running_loop().run_in_executor` to prevent blocking the event loop when dealing with file I/O operations.
