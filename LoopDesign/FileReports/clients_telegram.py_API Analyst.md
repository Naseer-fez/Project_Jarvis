# `telegram.py` - API Analyst Report

## Overview
Telegram integration via `python-telegram-bot` async Bot API to send and receive messages.

## Endpoints / Tools
1. `send_telegram`
   - Description: Send a Telegram message to the configured chat.
   - Risk: confirm (write)
   - Arguments: `message` (string, required), `parse_mode` (string, HTML or Markdown, default HTML).
2. `get_updates`
   - Description: Retrieve the latest incoming messages for the bot.
   - Risk: low (read-only)
   - Arguments: `limit` (integer, max 100, default 10).

## External Contracts / Dependencies
- Requires `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Uses `python-telegram-bot` package (`telegram.Bot`).

## Assumptions
- Uses an asynchronous context manager for the `telegram.Bot` to handle the underlying session.
- Messages are only sent to the strictly configured `TELEGRAM_CHAT_ID` rather than taking an arbitrary ID as an argument.
- Limit maxes out at 100 per `get_updates` call.
- Uses `parse_mode="HTML"` by default.
