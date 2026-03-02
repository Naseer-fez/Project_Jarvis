"""Email integration using stdlib smtplib + imaplib."""

from __future__ import annotations

import asyncio
import email as email_lib
import imaplib
import os
import smtplib
from email.mime.text import MIMEText
from typing import Any

from integrations.base import BaseIntegration


class EmailIntegration(BaseIntegration):
    name = "email"
    description = "Send and read emails via SMTP/IMAP"
    required_config: list[str] = ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "SMTP_HOST", "IMAP_HOST"]

    def is_available(self) -> bool:
        try:
            import smtplib as _smtplib  # noqa: F401
            import imaplib as _imaplib  # noqa: F401
            return all(bool(os.environ.get(key)) for key in self.required_config)
        except Exception:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_email",
                "description": "Send an email",
                "risk": "confirm",
                "args": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "read_emails",
                "description": "Read recent emails from inbox",
                "risk": "low",
                "args": {
                    "folder": {"type": "string", "default": "INBOX"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required_args": [],
            },
            {
                "name": "search_emails",
                "description": "Search emails by keyword",
                "risk": "low",
                "args": {
                    "query": {"type": "string", "description": "Subject keyword"},
                },
                "required_args": ["query"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            if tool_name == "send_email":
                data = await loop.run_in_executor(
                    None,
                    lambda: self._send_email(
                        to=str(args.get("to", "")),
                        subject=str(args.get("subject", "")),
                        body=str(args.get("body", "")),
                    ),
                )
                return {"success": True, "data": data, "error": None}

            if tool_name == "read_emails":
                folder = str(args.get("folder", "INBOX") or "INBOX")
                limit = max(1, int(args.get("limit", 10) or 10))
                data = await loop.run_in_executor(None, lambda: self._read_emails(folder=folder, limit=limit))
                return {"success": True, "data": {"emails": data}, "error": None}

            if tool_name == "search_emails":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"success": False, "data": None, "error": "query is required"}
                data = await loop.run_in_executor(None, lambda: self._search_emails(query=query))
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        if not to.strip():
            raise ValueError("to is required")

        addr = os.environ["EMAIL_ADDRESS"]
        pwd = os.environ["EMAIL_PASSWORD"]
        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = addr
        msg["To"] = to

        with smtplib.SMTP(host, port, timeout=10) as client:
            client.starttls()
            client.login(addr, pwd)
            client.send_message(msg)

        return {"sent_to": to}

    def _read_emails(self, folder: str = "INBOX", limit: int = 10) -> list[dict[str, Any]]:
        addr = os.environ["EMAIL_ADDRESS"]
        pwd = os.environ["EMAIL_PASSWORD"]
        host = os.environ["IMAP_HOST"]

        with imaplib.IMAP4_SSL(host, timeout=10) as client:
            client.login(addr, pwd)
            client.select(folder)
            _, data = client.search(None, "ALL")
            ids = (data[0] if data else b"").split()[-limit:]

            results: list[dict[str, Any]] = []
            for email_id in reversed(ids):
                _, fetched = client.fetch(email_id, "(RFC822)")
                if not fetched or not fetched[0]:
                    continue
                msg = email_lib.message_from_bytes(fetched[0][1])
                results.append(
                    {
                        "from": msg.get("From"),
                        "subject": msg.get("Subject"),
                        "date": msg.get("Date"),
                    }
                )

        return results

    def _search_emails(self, query: str) -> dict[str, Any]:
        addr = os.environ["EMAIL_ADDRESS"]
        pwd = os.environ["EMAIL_PASSWORD"]
        host = os.environ["IMAP_HOST"]
        safe_query = query.replace('"', "")

        with imaplib.IMAP4_SSL(host, timeout=10) as client:
            client.login(addr, pwd)
            client.select("INBOX")
            _, data = client.search(None, f'SUBJECT "{safe_query}"')
            ids = (data[0] if data else b"").split()
            return {
                "matches": len(ids),
                "ids": [item.decode("utf-8", errors="ignore") for item in ids[-10:]],
            }


__all__ = ["EmailIntegration"]
