"""Email integration (SMTP + IMAP) for Jarvis dynamic plugins."""

from __future__ import annotations

import asyncio
import email
import imaplib
import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency
    load_dotenv = None  # type: ignore[assignment]


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / "config" / "settings.env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)


def _get_config_value(config: Any, key: str, section: str = "email") -> str:
    env_key = key.upper()
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value

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


class EmailIntegration(BaseIntegration):
    name = "email"
    description = "Send and read email using SMTP/IMAP"
    required_config = [
        "email_address",
        "smtp_host",
        "smtp_port",
        "imap_host",
        "password",
    ]

    def __init__(self, config: Any = None) -> None:
        _load_env_file()
        self._config = config
        self.email_address = _get_config_value(config, "email_address")
        self.smtp_host = _get_config_value(config, "smtp_host")
        self.smtp_port = int(_get_config_value(config, "smtp_port") or "587")
        self.imap_host = _get_config_value(config, "imap_host")
        self.password = _get_config_value(config, "password")

    def is_available(self) -> bool:
        missing = [key for key in self.required_config if not getattr(self, key)]
        if missing:
            logger.warning("Email integration unavailable. Missing config keys: %s", ", ".join(missing))
            return False
        return True

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "send_email",
                "description": "Send an email via configured SMTP account.",
                "risk": "CONFIRM",
                "args": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject."},
                    "body": {"type": "string", "description": "Email body text."},
                },
                "required_args": ["to", "subject", "body"],
            },
            {
                "name": "read_emails",
                "description": "Read latest emails from inbox.",
                "risk": "LOW",
                "args": {
                    "limit": {"type": "integer", "description": "Max number of emails to return.", "default": 5},
                },
                "required_args": [],
            },
            {
                "name": "search_emails",
                "description": "Search inbox emails by IMAP query text.",
                "risk": "LOW",
                "args": {
                    "query": {"type": "string", "description": "Search query used against SUBJECT and BODY."},
                    "limit": {"type": "integer", "description": "Max number of results.", "default": 5},
                },
                "required_args": ["query"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool_name == "send_email":
                data = await asyncio.to_thread(
                    self._send_email,
                    str(args.get("to", "")).strip(),
                    str(args.get("subject", "")).strip(),
                    str(args.get("body", "")),
                )
                return {"success": True, "data": data, "error": None}

            if tool_name == "read_emails":
                limit = int(args.get("limit", 5) or 5)
                data = await asyncio.to_thread(self._read_emails, "ALL", limit)
                return {"success": True, "data": data, "error": None}

            if tool_name == "search_emails":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"success": False, "data": None, "error": "query is required"}
                limit = int(args.get("limit", 5) or 5)
                criteria = f'(OR SUBJECT "{query}" BODY "{query}")'
                data = await asyncio.to_thread(self._read_emails, criteria, limit)
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown email tool '{tool_name}'"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Email integration execution failed for %s", tool_name)
            return {"success": False, "data": None, "error": str(exc)}

    def _send_email(self, to_addr: str, subject: str, body: str) -> dict[str, Any]:
        if not to_addr:
            raise ValueError("Recipient address is required")

        msg = EmailMessage()
        msg["From"] = self.email_address
        msg["To"] = to_addr
        msg["Subject"] = subject or "(no subject)"
        msg.set_content(body or "")

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(self.email_address, self.password)
            smtp.send_message(msg)

        return {"to": to_addr, "subject": msg["Subject"], "status": "sent"}

    def _read_emails(self, criteria: str, limit: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        with imaplib.IMAP4_SSL(self.imap_host) as imap:
            imap.login(self.email_address, self.password)
            imap.select("INBOX")
            status, data = imap.search(None, criteria)
            if status != "OK" or not data:
                return items

            ids = data[0].split()[-max(limit, 1) :]
            for msg_id in reversed(ids):
                status, fetched = imap.fetch(msg_id, "(RFC822)")
                if status != "OK" or not fetched:
                    continue

                raw = fetched[0][1]
                parsed = email.message_from_bytes(raw)
                body = self._extract_text_body(parsed)
                items.append(
                    {
                        "id": msg_id.decode(errors="ignore"),
                        "from": str(parsed.get("From", "")),
                        "subject": str(parsed.get("Subject", "")),
                        "date": str(parsed.get("Date", "")),
                        "snippet": body[:200],
                    }
                )
        return items

    @staticmethod
    def _extract_text_body(msg: email.message.Message) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True) or b""
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
            return ""

        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")


__all__ = ["EmailIntegration"]
