"""WhatsApp integration using Twilio when configured."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency
    load_dotenv = None  # type: ignore[assignment]

try:
    from twilio.rest import Client as TwilioClient
except ImportError:  # optional dependency
    TwilioClient = None  # type: ignore[assignment]


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / "config" / "settings.env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)


def _cfg(config: Any, key: str, section: str = "whatsapp") -> str:
    value = os.environ.get(key.upper(), "").strip()
    if value:
        return value

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


class WhatsAppIntegration(BaseIntegration):
    name = "whatsapp"
    description = "Send and read WhatsApp messages via Twilio"
    required_config = [
        "twilio_account_sid",
        "twilio_auth_token",
        "twilio_whatsapp_from",
    ]

    def __init__(self, config: Any = None) -> None:
        _load_env_file()
        self._config = config
        self.account_sid = _cfg(config, "twilio_account_sid")
        self.auth_token = _cfg(config, "twilio_auth_token")
        self.from_number = _cfg(config, "twilio_whatsapp_from")

    def is_available(self) -> bool:
        if TwilioClient is None:
            logger.warning("WhatsApp integration unavailable: twilio package is not installed")
            return False

        if not self.account_sid:
            logger.warning("WhatsApp integration unavailable: TWILIO_ACCOUNT_SID is missing")
            return False

        if not self.auth_token or not self.from_number:
            logger.warning("WhatsApp integration unavailable: token/from number missing")
            return False

        return True

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "send_whatsapp",
                "description": "Send a WhatsApp message via Twilio sandbox/number.",
                "risk": "CONFIRM",
                "args": {
                    "to": {"type": "string", "description": "Recipient WhatsApp number like whatsapp:+123..."},
                    "message": {"type": "string", "description": "Message body."},
                },
                "required_args": ["to", "message"],
            },
            {
                "name": "read_whatsapp_messages",
                "description": "Read recent WhatsApp messages from Twilio account logs.",
                "risk": "LOW",
                "args": {
                    "limit": {"type": "integer", "description": "Maximum messages to return.", "default": 10},
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool_name == "send_whatsapp":
                to_value = str(args.get("to", "")).strip()
                message = str(args.get("message", ""))
                data = await asyncio.to_thread(self._send_whatsapp, to_value, message)
                return {"success": True, "data": data, "error": None}

            if tool_name == "read_whatsapp_messages":
                limit = int(args.get("limit", 10) or 10)
                data = await asyncio.to_thread(self._read_messages, limit)
                return {"success": True, "data": data, "error": None}

            return {"success": False, "data": None, "error": f"Unknown WhatsApp tool '{tool_name}'"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("WhatsApp integration execution failed for %s", tool_name)
            return {"success": False, "data": None, "error": str(exc)}

    def _client(self):
        if TwilioClient is None:
            raise RuntimeError("twilio package is not installed")
        return TwilioClient(self.account_sid, self.auth_token)

    def _send_whatsapp(self, to_number: str, message: str) -> dict[str, Any]:
        if not to_number:
            raise ValueError("'to' is required")
        if not message:
            raise ValueError("'message' is required")

        client = self._client()
        msg = client.messages.create(
            body=message,
            from_=self.from_number,
            to=to_number,
        )
        return {"sid": msg.sid, "status": msg.status, "to": to_number}

    def _read_messages(self, limit: int) -> list[dict[str, Any]]:
        client = self._client()
        items = client.messages.list(limit=max(limit, 1))
        results: list[dict[str, Any]] = []
        for item in items:
            from_value = str(getattr(item, "from_", "") or "")
            to_value = str(getattr(item, "to", "") or "")
            if "whatsapp:" not in from_value and "whatsapp:" not in to_value:
                continue

            results.append(
                {
                    "sid": item.sid,
                    "status": item.status,
                    "from": from_value,
                    "to": to_value,
                    "body": item.body,
                    "date_sent": str(item.date_sent),
                }
            )
        return results


__all__ = ["WhatsAppIntegration"]
