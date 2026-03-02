"""Google Calendar integration via Google Calendar API v3.

Uses fully async aiohttp for all HTTP calls. OAuth is handled via
refresh token — no browser popup at runtime.

Required env vars:
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    GOOGLE_REFRESH_TOKEN
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any

from integrations.base import BaseIntegration

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"


class GoogleCalendarIntegration(BaseIntegration):
    """Google Calendar v3 integration (async, RFC3339, OAuth refresh)."""

    name = "google_calendar"
    description = "Create, list, and delete Google Calendar events"
    required_config: list[str] = [
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REFRESH_TOKEN",
    ]

    def is_available(self) -> bool:
        try:
            import aiohttp  # noqa: F401
        except Exception:
            self.unavailable_reason = "aiohttp not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_event",
                "description": "Create a new event in Google Calendar",
                "risk": "confirm",
                "args": {
                    "summary": {"type": "string", "description": "Event title"},
                    "start": {
                        "type": "string",
                        "description": "Start datetime in ISO-8601 (e.g. 2026-03-15T10:00:00)",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime in ISO-8601",
                    },
                    "description": {
                        "type": "string",
                        "description": "Event description",
                        "default": "",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone (e.g. Asia/Kolkata)",
                        "default": "UTC",
                    },
                    "calendar_id": {
                        "type": "string",
                        "description": "Calendar ID (default: primary)",
                        "default": "primary",
                    },
                },
                "required_args": ["summary", "start", "end"],
            },
            {
                "name": "list_events",
                "description": "List upcoming Google Calendar events",
                "risk": "low",
                "args": {
                    "days_ahead": {"type": "integer", "description": "Look ahead N days", "default": 7},
                    "max_results": {"type": "integer", "description": "Max events to return", "default": 10},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": [],
            },
            {
                "name": "delete_event",
                "description": "Delete a Google Calendar event by its event ID",
                "risk": "confirm",
                "args": {
                    "event_id": {"type": "string", "description": "Google Calendar event ID"},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": ["event_id"],
            },
            {
                "name": "find_free_slot",
                "description": "Find the next free time slot of a given duration",
                "risk": "low",
                "args": {
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Required free slot duration in minutes",
                        "default": 60,
                    },
                    "days_ahead": {"type": "integer", "default": 7},
                    "calendar_id": {"type": "string", "default": "primary"},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            token = await self._refresh_access_token()
            if tool_name == "create_event":
                return await self._create_event(token, args)
            if tool_name == "list_events":
                return await self._list_events(token, args)
            if tool_name == "delete_event":
                return await self._delete_event(token, args)
            if tool_name == "find_free_slot":
                return await self._find_free_slot(token, args)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    async def _refresh_access_token(self) -> str:
        import aiohttp

        payload = {
            "client_id": os.environ["GOOGLE_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
            "refresh_token": os.environ["GOOGLE_REFRESH_TOKEN"],
            "grant_type": "refresh_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(_TOKEN_URL, data=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if "access_token" not in data:
                    raise RuntimeError(f"Token refresh failed: {data.get('error', 'unknown')}")
                return data["access_token"]

    def _to_rfc3339(self, dt_str: str, tz: str = "UTC") -> str:
        """Parse ISO-8601 string and return RFC3339 with timezone offset."""
        # Try with offset first, then naive
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(dt_str.strip(), fmt)
                if dt.tzinfo is None:
                    # Attach UTC; caller should pass correct tz string but we default safely
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {dt_str!r}")

    async def _create_event(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        summary = str(args.get("summary", "")).strip()
        if not summary:
            return {"success": False, "data": None, "error": "summary is required"}

        tz = str(args.get("timezone", "UTC") or "UTC")
        start_str = str(args.get("start", ""))
        end_str = str(args.get("end", ""))
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        event_body = {
            "summary": summary,
            "description": str(args.get("description", "") or ""),
            "start": {"dateTime": self._to_rfc3339(start_str), "timeZone": tz},
            "end": {"dateTime": self._to_rfc3339(end_str), "timeZone": tz},
        }

        url = f"{_CALENDAR_BASE}/calendars/{cal_id}/events"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=event_body, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                return {"success": True, "data": {"event_id": data["id"], "link": data.get("htmlLink")}, "error": None}

    async def _list_events(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        days_ahead = int(args.get("days_ahead", 7) or 7)
        max_results = min(50, int(args.get("max_results", 10) or 10))
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        now = datetime.now(tz=timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days_ahead)).isoformat()

        url = f"{_CALENDAR_BASE}/calendars/{cal_id}/events"
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                events = [
                    {
                        "id": item["id"],
                        "summary": item.get("summary", "Untitled"),
                        "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                        "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                    }
                    for item in data.get("items", [])
                ]
                return {"success": True, "data": {"events": events}, "error": None}

    async def _delete_event(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        event_id = str(args.get("event_id", "")).strip()
        if not event_id:
            return {"success": False, "data": None, "error": "event_id is required"}
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        url = f"{_CALENDAR_BASE}/calendars/{cal_id}/events/{event_id}"
        headers = {"Authorization": f"Bearer {token}"}

        async with aiohttp.ClientSession() as session:
            async with session.delete(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 204:
                    return {"success": True, "data": {"deleted": event_id}, "error": None}
                body = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": body.get("error", {}).get("message", str(resp.status)),
                }

    async def _find_free_slot(self, token: str, args: dict[str, Any]) -> dict[str, Any]:
        """Find the earliest free slot of the requested duration using freebusy query."""
        import aiohttp

        duration_min = int(args.get("duration_minutes", 60) or 60)
        days_ahead = int(args.get("days_ahead", 7) or 7)
        cal_id = str(args.get("calendar_id", "primary") or "primary")

        now = datetime.now(tz=timezone.utc)
        time_max = (now + timedelta(days=days_ahead)).isoformat()

        url = f"{_CALENDAR_BASE}/freeBusy"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        body = {
            "timeMin": now.isoformat(),
            "timeMax": time_max,
            "items": [{"id": cal_id}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }

        busy_slots = data.get("calendars", {}).get(cal_id, {}).get("busy", [])
        # Find first gap of >= duration_min minutes
        cursor = now
        for slot in busy_slots:
            busy_start = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
            gap = (busy_start - cursor).total_seconds() / 60
            if gap >= duration_min:
                return {
                    "success": True,
                    "data": {
                        "free_start": cursor.isoformat(),
                        "free_end": (cursor + timedelta(minutes=duration_min)).isoformat(),
                    },
                    "error": None,
                }
            busy_end = datetime.fromisoformat(slot["end"].replace("Z", "+00:00"))
            if busy_end > cursor:
                cursor = busy_end

        # Gap after all busy slots
        return {
            "success": True,
            "data": {
                "free_start": cursor.isoformat(),
                "free_end": (cursor + timedelta(minutes=duration_min)).isoformat(),
            },
            "error": None,
        }


__all__ = ["GoogleCalendarIntegration"]
