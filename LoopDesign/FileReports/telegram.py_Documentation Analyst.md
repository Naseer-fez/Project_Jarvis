# Documentation Report: clients/telegram.py

## Assumptions
- Uses `python-telegram-bot` (`telegram` module).
- Operates using long-polling for updates rather than webhooks (`get_updates` manual extraction).
- Targeted strictly at `TELEGRAM_CHAT_ID` so bot only replies or interacts with predefined user/group.

## Schema / API Contract
- Tools: `send_telegram(message, parse_mode)`, `get_updates(limit)`.
- Updates mapped to dictionary containing `update_id`, `from`, `text`, `date`.

## Dependencies
- `telegram` (external)
- `os` (stdlib)

## Configuration Variables
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Prompts
None.
