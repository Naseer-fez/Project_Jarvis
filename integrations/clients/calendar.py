"""Calendar integration with optional icalendar and local .ics fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

try:
    from icalendar import Calendar, Event
except Exception:  # noqa: BLE001
    Calendar = None  # type: ignore[assignment]
    Event = None  # type: ignore[assignment]


def _load_settings_env() -> None:
    env_file = Path(__file__).resolve().parents[2] / "config" / "settings.env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip())


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


def _parse_iso(text: str) -> datetime:
    value = (text or "").strip()
    if not value:
        raise ValueError("datetime value is required")

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ics_datetime(dt: datetime) -> str:
    as_utc = dt.astimezone(timezone.utc)
    return as_utc.strftime("%Y%m%dT%H%M%SZ")


class CalendarIntegration(BaseIntegration):
    name = "calendar"
    description = "Add and list events in a local calendar"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config=config)
        _load_settings_env()

        path_value = _cfg(config, "calendar_ics_path") or "data/calendar.ics"
        self.calendar_path = Path(path_value)
        self.calendar_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_calendar_file()

    def is_available(self) -> bool:
        # Local fallback means this integration can always run.
        self.unavailable_reason = ""
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "add_event",
                "description": "Add a calendar event to the local calendar file.",
                "risk": "CONFIRM",
                "args": {
                    "title": {"type": "string", "description": "Event title."},
                    "start": {"type": "string", "description": "Start datetime in ISO format."},
                    "end": {
                        "type": "string",
                        "description": "End datetime in ISO format. Defaults to start +1h.",
                        "default": "",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional details.",
                        "default": "",
                    },
                },
                "required_args": ["title", "start"],
            },
            {
                "name": "list_events",
                "description": "List calendar events.",
                "risk": "LOW",
                "args": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return.",
                        "default": 10,
                    }
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        loop = asyncio.get_running_loop()

        try:
            if tool_name == "add_event":
                title = str(args.get("title", "")).strip()
                start = str(args.get("start", "")).strip()
                end = str(args.get("end", "")).strip()
                description = str(args.get("description", ""))
                data = await loop.run_in_executor(None, self._add_event, title, start, end, description)
                return {"success": True, "data": data, "error": None}

            if tool_name == "list_events":
                limit = max(1, int(args.get("limit", 10) or 10))
                data = await loop.run_in_executor(None, self._list_events, limit)
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool '{tool_name}'"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Calendar execution failed for %s", tool_name)
            return {"success": False, "data": None, "error": str(exc)}

    def _ensure_calendar_file(self) -> None:
        if self.calendar_path.exists():
            return

        skeleton = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//Jarvis Calendar//EN\nEND:VCALENDAR\n"
        self.calendar_path.write_text(skeleton, encoding="utf-8")

    def _add_event(self, title: str, start: str, end: str, description: str) -> dict[str, Any]:
        if not title:
            raise ValueError("title is required")

        start_dt = _parse_iso(start)
        end_dt = _parse_iso(end) if end else (start_dt + timedelta(hours=1))
        uid = f"jarvis-{uuid.uuid4()}"

        if Calendar is not None and Event is not None:
            self._add_event_icalendar(uid, title, start_dt, end_dt, description)
        else:
            self._add_event_plain(uid, title, start_dt, end_dt, description)

        return {
            "uid": uid,
            "title": title,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "calendar_file": str(self.calendar_path),
        }

    def _add_event_icalendar(
        self,
        uid: str,
        title: str,
        start_dt: datetime,
        end_dt: datetime,
        description: str,
    ) -> None:
        raw = self.calendar_path.read_bytes() if self.calendar_path.exists() else b""
        calendar = Calendar.from_ical(raw) if raw else Calendar()
        if not raw:
            calendar.add("prodid", "-//Jarvis Calendar//EN")
            calendar.add("version", "2.0")

        event = Event()
        event.add("uid", uid)
        event.add("summary", title)
        event.add("dtstart", start_dt)
        event.add("dtend", end_dt)
        if description:
            event.add("description", description)

        calendar.add_component(event)
        self.calendar_path.write_bytes(calendar.to_ical())

    def _add_event_plain(
        self,
        uid: str,
        title: str,
        start_dt: datetime,
        end_dt: datetime,
        description: str,
    ) -> None:
        text = self.calendar_path.read_text(encoding="utf-8", errors="replace").rstrip()
        if text.endswith("END:VCALENDAR"):
            text = text[:-len("END:VCALENDAR")].rstrip()

        block = (
            "BEGIN:VEVENT\n"
            f"UID:{uid}\n"
            f"DTSTART:{_ics_datetime(start_dt)}\n"
            f"DTEND:{_ics_datetime(end_dt)}\n"
            f"SUMMARY:{title}\n"
            f"DESCRIPTION:{description}\n"
            "END:VEVENT\n"
            "END:VCALENDAR\n"
        )
        self.calendar_path.write_text(text + "\n" + block, encoding="utf-8")

    def _list_events(self, limit: int) -> list[dict[str, Any]]:
        if Calendar is not None:
            return self._list_events_icalendar(limit)
        return self._list_events_plain(limit)

    def _list_events_icalendar(self, limit: int) -> list[dict[str, Any]]:
        raw = self.calendar_path.read_bytes() if self.calendar_path.exists() else b""
        if not raw:
            return []

        calendar = Calendar.from_ical(raw)
        events: list[dict[str, Any]] = []
        for component in calendar.walk():
            if getattr(component, "name", "") != "VEVENT":
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
        return events[:limit]

    def _list_events_plain(self, limit: int) -> list[dict[str, Any]]:
        text = self.calendar_path.read_text(encoding="utf-8", errors="replace")
        blocks = re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, flags=re.DOTALL)

        def _pick(block: str, field: str) -> str:
            match = re.search(rf"{field}:(.*)", block)
            return match.group(1).strip() if match else ""

        events: list[dict[str, Any]] = []
        for block in blocks:
            events.append(
                {
                    "uid": _pick(block, "UID"),
                    "title": _pick(block, "SUMMARY"),
                    "start": _pick(block, "DTSTART"),
                    "end": _pick(block, "DTEND"),
                    "description": _pick(block, "DESCRIPTION"),
                }
            )

        events.sort(key=lambda item: item.get("start", ""))
        return events[:limit]


__all__ = ["CalendarIntegration"]
