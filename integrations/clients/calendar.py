"""Local calendar integration backed by a simple .ics file."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

CALENDAR_PATH = Path("memory/calendar.ics")


class CalendarIntegration(BaseIntegration):
    name = "calendar"
    description = "Manage a local calendar (.ics file)"
    required_config: list[str] = []

    def is_available(self) -> bool:
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "add_event",
                "description": "Add an event to calendar",
                "risk": "confirm",
                "args": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM", "default": "09:00"},
                    "duration_minutes": {"type": "integer", "default": 60},
                },
                "required_args": ["title", "date"],
            },
            {
                "name": "list_events",
                "description": "List upcoming events",
                "risk": "low",
                "args": {
                    "days_ahead": {"type": "integer", "default": 7},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        try:
            if tool_name == "add_event":
                data = await loop.run_in_executor(
                    None,
                    lambda: self._add_event(
                        title=str(args.get("title", "")),
                        date=str(args.get("date", "")),
                        time=str(args.get("time", "09:00") or "09:00"),
                        duration_minutes=int(args.get("duration_minutes", 60) or 60),
                    ),
                )
                return {"success": True, "data": data, "error": None}

            if tool_name == "list_events":
                days_ahead = int(args.get("days_ahead", 7) or 7)
                data = await loop.run_in_executor(None, lambda: self._list_events(days_ahead=days_ahead))
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _add_event(
        self,
        title: str,
        date: str,
        time: str = "09:00",
        duration_minutes: int = 60,
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValueError("title is required")
        if not date.strip():
            raise ValueError("date is required")

        dt_start = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        dt_end = dt_start + timedelta(minutes=duration_minutes)
        uid = f"{dt_start.strftime('%Y%m%dT%H%M%S')}-jarvis"

        block = (
            "BEGIN:VEVENT\n"
            f"DTSTART:{dt_start.strftime('%Y%m%dT%H%M%S')}\n"
            f"DTEND:{dt_end.strftime('%Y%m%dT%H%M%S')}\n"
            f"SUMMARY:{title}\n"
            f"UID:{uid}\n"
            "END:VEVENT\n"
        )

        if not CALENDAR_PATH.exists():
            CALENDAR_PATH.write_text("BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n", encoding="utf-8")

        content = CALENDAR_PATH.read_text(encoding="utf-8")
        updated = content.replace("END:VCALENDAR", block + "END:VCALENDAR")
        CALENDAR_PATH.write_text(updated, encoding="utf-8")

        return {"event": title, "date": date, "time": time}

    def _list_events(self, days_ahead: int = 7) -> dict[str, Any]:
        if not CALENDAR_PATH.exists():
            return {"events": []}
        
        from icalendar import Calendar
        from dateutil.tz import tzlocal
        import datetime as dt
        
        cal = Calendar.from_ical(CALENDAR_PATH.read_bytes())
        now = dt.datetime.now(tz=tzlocal())
        cutoff = now + dt.timedelta(days=days_ahead)
        events = []
        
        for component in cal.walk():
            if component.name != "VEVENT":
                continue
            dtstart = component.get("DTSTART")
            if dtstart is None:
                continue
            start = dtstart.dt
            # Handle date-only events
            if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
                start = dt.datetime(start.year, start.month, start.day, tzinfo=tzlocal())
            if start.tzinfo is None:
                start = start.replace(tzinfo=tzlocal())
            if now <= start <= cutoff:
                events.append({
                    "title": str(component.get("SUMMARY", "Untitled")),
                    "datetime": str(start),
                })
        
        return {"events": sorted(events, key=lambda e: e["datetime"])}


__all__ = ["CalendarIntegration"]
