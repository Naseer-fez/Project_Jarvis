"""Telegram integration via python-telegram-bot async Bot API.

Required env vars:
    TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
    TELEGRAM_CHAT_ID    — Target chat ID (your personal or group chat)
"""

from __future__ import annotations

import os
from typing import Any

from integrations.base import BaseIntegration


class TelegramIntegration(BaseIntegration):
    """Send messages and receive updates via a Telegram bot."""

    name = "telegram"
    description = "Send and receive Telegram messages via a bot"
    required_config: list[str] = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

    def is_available(self) -> bool:
        try:
            import telegram  # noqa: F401
        except Exception:
            self.unavailable_reason = "python-telegram-bot not installed"
            return False
        if not all(bool(os.environ.get(k)) for k in self.required_config):
            missing = [k for k in self.required_config if not os.environ.get(k)]
            self.unavailable_reason = f"Missing env vars: {missing}"
            return False
        return True

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_telegram",
                "description": "Send a Telegram message to the configured chat",
                "risk": "confirm",
                "args": {
                    "message": {
                        "type": "string",
                        "description": "The text message to send",
                    },
                    "parse_mode": {
                        "type": "string",
                        "description": "Optional: HTML or Markdown",
                        "default": "HTML",
                    },
                },
                "required_args": ["message"],
            },
            {
                "name": "get_updates",
                "description": "Retrieve the latest incoming messages for the bot",
                "risk": "low",
                "args": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of updates to fetch (max 100)",
                        "default": 10,
                    },
                },
                "required_args": [],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "send_telegram":
                return await self._send_telegram(
                    message=str(args.get("message", "")),
                    parse_mode=str(args.get("parse_mode", "HTML") or "HTML"),
                )
            if tool_name == "get_updates":
                limit = max(1, min(100, int(args.get("limit", 10) or 10)))
                return await self._get_updates(limit=limit)
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "data": None, "error": str(exc)}

    async def _send_telegram(self, message: str, parse_mode: str = "HTML") -> dict[str, Any]:
        if not message.strip():
            return {"success": False, "data": None, "error": "message is required"}

        from telegram import Bot

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]

        bot = Bot(token=token)
        async with bot:
            sent = await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
            )
        return {
            "success": True,
            "data": {"message_id": sent.message_id, "chat_id": str(chat_id)},
            "error": None,
        }

    async def _get_updates(self, limit: int = 10) -> dict[str, Any]:
        from telegram import Bot

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        bot = Bot(token=token)
        async with bot:
            updates = await bot.get_updates(limit=limit)

        messages = []
        for update in updates:
            if update.message:
                messages.append(
                    {
                        "update_id": update.update_id,
                        "from": update.message.from_user.username if update.message.from_user else None,
                        "text": update.message.text or "",
                        "date": str(update.message.date),
                    }
                )
        return {"success": True, "data": {"updates": messages}, "error": None}


__all__ = ["TelegramIntegration"]
