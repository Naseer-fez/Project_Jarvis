"""Email integration using only stdlib smtplib + imaplib."""

from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import os
import smtplib
from email.header import decode_header
from email.message import EmailMessage, Message
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


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


def _cfg(config: Any, key: str, section: str = "email") -> str:
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


def _decode_header_value(value: str) -> str:
    if not value:
        return ""

    chunks: list[str] = []
    for part, enc in decode_header(value):
        if isinstance(part, bytes):
            chunks.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            chunks.append(str(part))
    return "".join(chunks)


class EmailIntegration(BaseIntegration):
    name = "email"
    description = "Send, read, and search emails over SMTP/IMAP"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config=config)
        _load_settings_env()

        self.email_address = _cfg(config, "email_address")
        self.smtp_host = _cfg(config, "smtp_host")
        self.smtp_port = int(_cfg(config, "smtp_port") or "587")
        self.imap_host = _cfg(config, "imap_host")
        self.password = _cfg(config, "password")

    def is_available(self) -> bool:
        missing: list[str] = []
        if not self.email_address:
            missing.append("EMAIL_ADDRESS")
        if not self.smtp_host:
            missing.append("SMTP_HOST")
        if not self.imap_host:
            missing.append("IMAP_HOST")
        if not self.password:
            missing.append("PASSWORD")

        if missing:
            self.unavailable_reason = f"Missing required config/env: {', '.join(missing)}"
            return False

        self.unavailable_reason = ""
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_email",
                "description": "Send an email using configured SMTP account.",
                "risk": "CONFIRM",
                "args": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Message subject."},
                    "body": {"type": "string", "description": "Message body."},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "read_emails",
                "description": "Read recent inbox emails.",
                "risk": "LOW",
                "args": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items to return.",
                        "default": 5,
                    }
                },
                "required_args": [],
            },
            {
                "name": "search_emails",
                "description": "Search inbox emails by text query.",
                "risk": "LOW",
                "args": {
                    "query": {"type": "string", "description": "Search text."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items to return.",
                        "default": 5,
                    },
                },
                "required_args": ["query"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        loop = asyncio.get_running_loop()

        try:
            if tool_name == "send_email":
                to_addr = str(args.get("to", "")).strip()
                subject = str(args.get("subject", "")).strip()
                body = str(args.get("body", ""))
                data = await loop.run_in_executor(None, self._send_email, to_addr, subject, body)
                return {"success": True, "data": data, "error": None}

            if tool_name == "read_emails":
                limit = max(1, int(args.get("limit", 5) or 5))
                data = await loop.run_in_executor(None, self._read_emails, "ALL", limit)
                return {"success": True, "data": data, "error": None}

            if tool_name == "search_emails":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"success": False, "data": None, "error": "query is required"}

                limit = max(1, int(args.get("limit", 5) or 5))
                sanitized = query.replace('"', " ")
                criteria = f'(TEXT "{sanitized}")'
                data = await loop.run_in_executor(None, self._read_emails, criteria, limit)
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown tool '{tool_name}'"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Email integration execution failed for %s", tool_name)
            return {"success": False, "data": None, "error": str(exc)}

    def _send_email(self, to_addr: str, subject: str, body: str) -> dict[str, Any]:
        if not to_addr:
            raise ValueError("'to' is required")

        msg = EmailMessage()
        msg["From"] = self.email_address
        msg["To"] = to_addr
        msg["Subject"] = subject or "(no subject)"
        msg.set_content(body or "")

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(self.email_address, self.password)
            smtp.send_message(msg)

        return {
            "to": to_addr,
            "subject": msg["Subject"],
            "status": "sent",
        }

    def _read_emails(self, criteria: str, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        with imaplib.IMAP4_SSL(self.imap_host) as imap:
            imap.login(self.email_address, self.password)
            imap.select("INBOX")

            status, data = imap.search(None, criteria)
            if status != "OK" or not data or not data[0]:
                return items

            ids = data[0].split()
            if not ids:
                return items

            selected_ids = ids[-limit:]
            for msg_id in reversed(selected_ids):
                status, fetched = imap.fetch(msg_id, "(RFC822)")
                if status != "OK" or not fetched:
                    continue

                raw = fetched[0][1]
                if not isinstance(raw, (bytes, bytearray)):
                    continue

                parsed = email.message_from_bytes(raw)
                body = self._extract_body(parsed)
                items.append(
                    {
                        "id": msg_id.decode(errors="ignore"),
                        "from": _decode_header_value(str(parsed.get("From", ""))),
                        "subject": _decode_header_value(str(parsed.get("Subject", ""))),
                        "date": str(parsed.get("Date", "")),
                        "snippet": body[:240],
                    }
                )

        return items

    @staticmethod
    def _extract_body(message: Message) -> str:
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type != "text/plain":
                    continue

                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
            return ""

        payload = message.get_payload(decode=True)
        if payload is None:
            return ""
        charset = message.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")


__all__ = ["EmailIntegration"]
