# File Report: telegram.py
## Role: Dependency Analyst

### 1. Library Requirements
- `os`, `typing` (Standard Library)
- `telegram` (Third-party, specifically `python-telegram-bot`)
- `integrations.base` (Local)

### 2. Service Dependencies
- Telegram Bot API.

### 3. Hidden Execution Links
- Acts directly via the `Bot` class from `python-telegram-bot` in an async context manager (`async with bot:`).
- `get_updates` explicitly pulls messages and constructs a clean summary rather than returning raw Telegram Update objects.

### 4. Assumptions & API Contracts
- Assumes the Bot token is valid and active (`@BotFather`).
- Parse mode defaults to `"HTML"`. Can be customized but assumes valid markup.
- Limits fetched updates to between 1 and 100 via `max(1, min(100, int(args.get("limit", 10) or 10)))`.
- Assumes a single configured target chat ID is sufficient for sending messages (`TELEGRAM_CHAT_ID`).

### 5. Configuration Variables
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### 6. Prompts Found
- None.
