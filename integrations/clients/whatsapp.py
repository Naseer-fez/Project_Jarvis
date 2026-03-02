"""WhatsApp integration via Twilio (optional dependency)."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)

try:
    from twilio.rest import Client as TwilioClient
except Exception:  # noqa: BLE001
    TwilioClient = None  # type: ignore[assignment]


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


def _cfg(config: Any, key: str, section: str = "whatsapp") -> str:
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


class WhatsAppIntegration(BaseIntegration):
    name = "whatsapp"
    description = "Send WhatsApp messages through Twilio"

    def __init__(self, config: Any = None) -> None:
        super().__init__(config=config)
        _load_settings_env()

        self.account_sid = _cfg(config, "twilio_account_sid")
        self.auth_token = _cfg(config, "twilio_auth_token")
        self.from_number = _cfg(config, "twilio_whatsapp_from")

    def is_available(self) -> bool:
        if TwilioClient is None:
            self.unavailable_reason = "twilio package not installed"
            return False
        if not self.account_sid:
            self.unavailable_reason = "TWILIO_ACCOUNT_SID missing"
            return False
        if not self.auth_token:
            self.unavailable_reason = "TWILIO_AUTH_TOKEN missing"
            return False
        if not self.from_number:
            self.unavailable_reason = "TWILIO_WHATSAPP_FROM missing"
            return False

        self.unavailable_reason = ""
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_whatsapp",
                "description": "Send a WhatsApp message via Twilio.",
                "risk": "CONFIRM",
                "args": {
                    "to": {
                        "type": "string",
                        "description": "Recipient in whatsapp:+123456 format.",
                    },
                    "message": {"type": "string", "description": "Message body."},
                },
                "required_args": ["to", "message"],
            }
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "send_whatsapp":
            return {"success": False, "data": None, "error": f"Unknown tool '{tool_name}'"}

        to_number = str((args or {}).get("to", "")).strip()
        message = str((args or {}).get("message", ""))

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, self._send_message, to_number, message)
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            logger.exception("WhatsApp send failed")
            return {"success": False, "data": None, "error": str(exc)}

    def _build_client(self):
        if TwilioClient is None:
            raise RuntimeError("twilio package not installed")
        return TwilioClient(self.account_sid, self.auth_token)

    def _send_message(self, to_number: str, message: str) -> dict[str, Any]:
        if not to_number:
            raise ValueError("'to' is required")
        if not message:
            raise ValueError("'message' is required")

        client = self._build_client()
        response = client.messages.create(body=message, from_=self.from_number, to=to_number)
        return {
            "sid": str(getattr(response, "sid", "")),
            "status": str(getattr(response, "status", "")),
            "to": to_number,
        }


__all__ = ["WhatsAppIntegration"]
