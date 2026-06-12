# clients/telegram.py API Analyst Report

## Overview
Telegram bot integration for sending and receiving messages.

## API Contracts & Methods
- `TelegramIntegration(BaseIntegration)`
  - Uses `python-telegram-bot` (`telegram.Bot`) for async operations.

## Tools Exposed
- `send_telegram(message, parse_mode="HTML")` [Risk: `confirm`]
- `get_updates(limit=10)` [Risk: `low`]

## Configuration Variables
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Assumptions & Constants
- The target chat ID is statically configured in environment variables, meaning the bot will primarily interact with a single user/group by default.

## Dependencies
- `python-telegram-bot` (`telegram`)

## Prompts
- None.
