"""Calendar integration using local ICS storage with optional icalendar library."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency
    load_dotenv = None  # type: ignore[assignment]

try:
    from icalendar import Calendar, Event
except ImportError:  # optional dependency
    Calendar = None  # type: ignore[assignment]
    Event = None  # type: ignore[assignment]


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / "config" / "settings.env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)


def _cfg(config: Any, key: str, section: str = "calendar") -> str:
    env_val = os.environ.get(key.upper(), "").strip()
    if env_val:
        return env_val

    if config is None:
        return ""

    try:
        if hasattr(config, "has_option") and config.has_option(section, key):
            return str(config.get(section, key)).strip()
        if hasattr(config, "has_option") and config.has_option("integrations", key):
            return str(config.get("integrations", key)).strip()
    except Exception:  # noqa: BLE001
        return ""
    return ""


class CalendarIntegration(BaseIntegration):
    name = "calendar"
    description = "Manage calendar events via local ICS file"
    required_config = []

    def __init__(self, config: Any = None) -> None:
        _load_env_file()
        self._config = config
        path_value = _cfg(config, "calendar_ics_path") or "data/calendar.ics"
        self.ics_path = Path(path_value)
        self.ics_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_calendar_file()

    def is_available(self) -> bool:
        # Always available with local ICS fallback.
        return True

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "add_calendar_event",
                "description": "Add an event to the local calendar.",
                "risk": "CONFIRM",
                "args": {
                    "title": {"type": "string", "description": "Event title."},
                    "start": {"type": "string", "description": "ISO datetime for start time."},
                    "end": {"type": "string", "description": "ISO datetime for end time.", "default": ""},
                    "description": {"type": "string", "description": "Optional event description.", "default": ""},
                },
                "required_args": ["title", "start"],
            },
            {
                "name": "list_calendar_events",
                "description": "List calendar events.",
                "risk": "LOW",
                "args": {
                    "limit": {"type": "integer", "description": "Max events to return.", "default": 10},
                },
                "required_args": [],
            },
            {
                "name": "search_calendar",
                "description": "Search calendar events by keyword.",
                "risk": "LOW",
                "args": {
                    "query": {"type": "string", "description": "Search text for title/description."},
                    "limit": {"type": "integer", "description": "Max events to return.", "default": 10},
                },
                "required_args": ["query"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool_name == "add_calendar_event":
                data = await asyncio.to_thread(
                    self._add_event,
                    str(args.get("title", "")).strip(),
                    str(args.get("start", "")).strip(),
                    str(args.get("end", "")).strip(),
                    str(args.get("description", "")),
                )
                return {"success": True, "data": data, "error": None}

            if tool_name == "list_calendar_events":
                limit = int(args.get("limit", 10) or 10)
                data = await asyncio.to_thread(self._list_events, limit)
                return {"success": True, "data": data, "error": None}

            if tool_name == "search_calendar":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"success": False, "data": None, "error": "query is required"}
                limit = int(args.get("limit", 10) or 10)
                data = await asyncio.to_thread(self._search_events, query, limit)
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown calendar tool '{tool_name}'"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Calendar integration execution failed for %s", tool_name)
            return {"success": False, "data": None, "error": str(exc)}

    def _ensure_calendar_file(self) -> None:
        if self.ics_path.exists():
            return

        if Calendar is not None:
            cal = Calendar()
            cal.add("prodid", "-//Jarvis Calendar//EN")
            cal.add("version", "2.0")
            self.ics_path.write_bytes(cal.to_ical())
            return

        self.ics_path.write_text(
            "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Jarvis Calendar//EN\nEND:VCALENDAR\n",
            encoding="utf-8",
        )

    def _add_event(self, title: str, start: str, end: str, description: str) -> dict[str, Any]:
        if not title:
            raise ValueError("title is required")
        if not start:
            raise ValueError("start is required")

        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end) if end else start_dt + timedelta(hours=1)
        uid = f"jarvis-{uuid.uuid4()}"

        if Calendar is not None and Event is not None:
            cal = Calendar.from_ical(self.ics_path.read_bytes())
            event = Event()
            event.add("uid", uid)
            event.add("summary", title)
            event.add("dtstart", start_dt)
            event.add("dtend", end_dt)
            if description:
                event.add("description", description)
            cal.add_component(event)
            self.ics_path.write_bytes(cal.to_ical())
        else:
            self._append_plain_ics_event(uid, title, start_dt, end_dt, description)

        return {
            "uid": uid,
            "title": title,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "calendar_file": str(self.ics_path),
        }

    def _list_events(self, limit: int) -> list[dict[str, Any]]:
        if Calendar is not None:
            cal = Calendar.from_ical(self.ics_path.read_bytes())
            events = []
            for component in cal.walk():
                if component.name != "VEVENT":
                    continue
                events.append(
                    {
                        "uid": str(component.get("uid", "")),
                        "title": str(component.get("summary", "")),
                        "start": str(component.get("dtstart", "")),
                        "end": str(component.get("dtend", "")),
                        "description": str(component.get("description", "")),
                    }
                )
            events.sort(key=lambda item: item.get("start", ""))
            return events[: max(limit, 1)]

        return self._list_events_plain(limit)

    def _search_events(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_l = query.lower()
        events = self._list_events(limit=1000)
        filtered = [
            item
            for item in events
            if query_l in item.get("title", "").lower()
            or query_l in item.get("description", "").lower()
        ]
        return filtered[: max(limit, 1)]

    def _append_plain_ics_event(
        self,
        uid: str,
        title: str,
        start_dt: datetime,
        end_dt: datetime,
        description: str,
    ) -> None:
        content = self.ics_path.read_text(encoding="utf-8", errors="replace")
        content = content.rstrip()
        if content.endswith("END:VCALENDAR"):
            content = content[: -len("END:VCALENDAR")].rstrip()

        event_block = (
            "BEGIN:VEVENT\n"
            f"UID:{uid}\n"
            f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}\n"
            f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}\n"
            f"SUMMARY:{title}\n"
            f"DESCRIPTION:{description}\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        self.ics_path.write_text(content + "\n" + event_block, encoding="utf-8")

    def _list_events_plain(self, limit: int) -> list[dict[str, Any]]:
        text = self.ics_path.read_text(encoding="utf-8", errors="replace")
        blocks = re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.DOTALL)
        events: list[dict[str, Any]] = []
        for block in blocks:
            def _pick(field: str) -> str:
                match = re.search(rf"{field}:(.*)", block)
                return match.group(1).strip() if match else ""

            events.append(
                {
                    "uid": _pick("UID"),
                    "title": _pick("SUMMARY"),
                    "start": _pick("DTSTART"),
                    "end": _pick("DTEND"),
                    "description": _pick("DESCRIPTION"),
                }
            )
        events.sort(key=lambda item: item.get("start", ""))
        return events[: max(limit, 1)]


__all__ = ["CalendarIntegration"]
