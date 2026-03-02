"""Gmail integration via Gmail API v1 (async aiohttp).

Uses the same Google OAuth credentials as google_calendar.py.

Required env vars:
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    GOOGLE_REFRESH_TOKEN

Rules:
- Email content is ALWAYS truncated to 2 000 chars before any LLM injection
- summarize_unread uses task_type="synthesis"
- No raw email headers injected blindly into context
"""

from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from typing import Any

from integrations.base import BaseIntegration

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_MAX_BODY_CHARS = 2000  # Truncation guard before any LLM injection


class GmailIntegration(BaseIntegration):
    """Gmail API v1 integration — async, token-refreshing, truncation-safe."""

    name = "gmail"
    description = "Read, send, and manage Gmail messages"
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
                "name": "list_unread",
                "description": "List unread emails from Gmail inbox",
                "risk": "low",
                "args": {
                    "max_results": {
                        "type": "integer",
                        "description": "Max emails to return",
                        "default": 10,
                    },
                },
                "required_args": [],
            },
            {
                "name": "send_gmail",
                "description": "Send an email via Gmail",
                "risk": "confirm",
                "args": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Plain-text email body"},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "summarize_unread",
                "description": "Fetch unread emails and return truncated content for LLM summarization",
                "risk": "low",
                "args": {
                    "max_results": {"type": "integer", "default": 5},
                },
                "required_args": [],
            },
            {
                "name": "mark_as_read",
                "description": "Mark a Gmail message as read by its message ID",
                "risk": "confirm",
                "args": {
                    "message_id": {
                        "type": "string",
                        "description": "Gmail message ID",
                    },
                },
                "required_args": ["message_id"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            token = await self._refresh_access_token()
            if tool_name == "list_unread":
                return await self._list_unread(token, int(args.get("max_results", 10) or 10))
            if tool_name == "send_gmail":
                return await self._send_gmail(
                    token,
                    to=str(args.get("to", "")),
                    subject=str(args.get("subject", "")),
                    body=str(args.get("body", "")),
                )
            if tool_name == "summarize_unread":
                return await self._summarize_unread(token, int(args.get("max_results", 5) or 5))
            if tool_name == "mark_as_read":
                return await self._mark_as_read(token, str(args.get("message_id", "")))
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    # ── OAuth ─────────────────────────────────────────────────────────────────

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

    # ── Tool implementations ──────────────────────────────────────────────────

    async def _list_unread(self, token: str, max_results: int) -> dict[str, Any]:
        import aiohttp

        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": "is:unread", "maxResults": min(max_results, 50)}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_GMAIL_BASE}/messages",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }

            messages = data.get("messages", [])
            summaries = []
            for m in messages[:max_results]:
                meta = await self._get_message_meta(session, headers, m["id"])
                summaries.append(meta)

        return {"success": True, "data": {"unread": summaries, "total": data.get("resultSizeEstimate", 0)}, "error": None}

    async def _get_message_meta(self, session: Any, headers: dict, message_id: str) -> dict[str, Any]:
        async with session.get(
            f"{_GMAIL_BASE}/messages/{message_id}",
            headers=headers,
            params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json()

        header_map: dict[str, str] = {}
        for h in data.get("payload", {}).get("headers", []):
            header_map[h["name"]] = h["value"]

        return {
            "id": message_id,
            "from": header_map.get("From", ""),
            "subject": header_map.get("Subject", "")[:200],  # truncate subject
            "date": header_map.get("Date", ""),
            "snippet": data.get("snippet", "")[:_MAX_BODY_CHARS],
        }

    async def _summarize_unread(self, token: str, max_results: int) -> dict[str, Any]:
        """Return truncated email content safe for LLM summarization."""
        result = await self._list_unread(token, max_results)
        if not result["success"]:
            return result
        # Content is already truncated in _get_message_meta; add metadata
        emails = result["data"]["unread"]
        safe_content = [
            {
                "from": e["from"],
                "subject": e["subject"],
                "snippet": e["snippet"][:_MAX_BODY_CHARS],
                "id": e["id"],
            }
            for e in emails
        ]
        return {
            "success": True,
            "data": {
                "emails_for_summary": safe_content,
                "count": len(safe_content),
                "task_type": "synthesis",  # hint for LLM router
            },
            "error": None,
        }

    async def _send_gmail(self, token: str, to: str, subject: str, body: str) -> dict[str, Any]:
        import aiohttp

        if not to.strip():
            return {"success": False, "data": None, "error": "to is required"}

        msg = MIMEText(body, "plain")
        msg["To"] = to
        msg["Subject"] = subject
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"raw": raw}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_GMAIL_BASE}/messages/send",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    return {
                        "success": False,
                        "data": None,
                        "error": data.get("error", {}).get("message", str(resp.status)),
                    }
                return {"success": True, "data": {"message_id": data.get("id")}, "error": None}

    async def _mark_as_read(self, token: str, message_id: str) -> dict[str, Any]:
        import aiohttp

        if not message_id.strip():
            return {"success": False, "data": None, "error": "message_id is required"}

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"removeLabelIds": ["UNREAD"]}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_GMAIL_BASE}/messages/{message_id}/modify",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return {"success": True, "data": {"marked_read": message_id}, "error": None}
                data = await resp.json()
                return {
                    "success": False,
                    "data": None,
                    "error": data.get("error", {}).get("message", str(resp.status)),
                }


__all__ = ["GmailIntegration"]
