"""WhatsApp integration via Twilio."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from integrations.base import BaseIntegration


class WhatsAppIntegration(BaseIntegration):
    name = "whatsapp"
    description = "Send WhatsApp messages via Twilio"
    required_config: list[str] = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]

    def is_available(self) -> bool:
        try:
            import twilio  # noqa: F401
            return all(bool(os.environ.get(key)) for key in self.required_config)
        except Exception:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_whatsapp",
                "description": "Send a WhatsApp message",
                "risk": "confirm",
                "args": {
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number with country code",
                    },
                    "message": {"type": "string", "description": "Message body"},
                },
                "required_args": ["to", "message"],
            }
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "send_whatsapp":
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}

        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None,
                lambda: self._send_whatsapp(
                    to=str(args.get("to", "")),
                    message=str(args.get("message", "")),
                ),
            )
            return {"success": True, "data": data, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    def _send_whatsapp(self, to: str, message: str) -> dict[str, Any]:
        if not to.strip():
            raise ValueError("to is required")
        if not message:
            raise ValueError("message is required")

        from twilio.rest import Client

        client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        msg = client.messages.create(
            from_=os.environ["TWILIO_WHATSAPP_FROM"],
            to=f"whatsapp:{to}",
            body=message,
        )
        return {"sid": str(msg.sid)}


__all__ = ["WhatsAppIntegration"]
