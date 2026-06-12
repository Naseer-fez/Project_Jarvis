import os
from pathlib import Path

reports_dir = Path(r"d:\AI\Jarvis\LoopDesign\FileReports")
reports_dir.mkdir(parents=True, exist_ok=True)

reports = {
    "base.py": """# Documentation Report: base.py

## Assumptions
- Every integration inherits from `BaseIntegration`
- `is_available` returns boolean natively without raising exceptions; failure reasons are populated in `self.unavailable_reason`
- Tools are declared in `get_tools()` returning a list of dicts mapping to `SYSTEM_TOOL_SCHEMA` format
- `execute` is asynchronous and returns an `IntegrationResult` compatible dictionary: `{"success": bool, "data": Any, "error": str | None}`

## Schema / API Contract
- `IntegrationResult`, `ToolResult`, `IntegrationRiskLevel`, `RiskLevel` are imported from `core.types.common`
- `name` (str), `description` (str), `required_config` (list[str]) are class variables defining the integration.
- Initialization accepts an optional `config` object.

## Dependencies
- `abc` (stdlib)
- `typing` (stdlib)
- `core.types.common` (internal)

## Configuration Variables
- No direct env vars, configuration is assumed to be passed during init or handled by children.

## Prompts
None.
""",

    "loader.py": """# Documentation Report: loader.py

## Assumptions
- Integrations reside in `integrations.clients` package
- Any `.py` file not prefixed with `_` is a potential integration module.
- It iterates through all modules inside `clients/`, uses `importlib` to import them, finds subclasses of `BaseIntegration`, and instantiates them.
- Availability is checked before registering in `registry`.
- Exceptions during import, instantiation, or registration are caught, logged, and skipped to allow gracefully failing.
- Callers may pass a config, but integrations currently use environment variables.

## Schema / API Contract
- `IntegrationLoader.load_all(config: Any, registry: Any) -> dict[str, list[str]]` where returned dictionary has `loaded` and `skipped` lists of integration names.
- `load_all` top-level function wraps `IntegrationLoader().load_all`.

## Dependencies
- `importlib`, `inspect`, `logging`, `pathlib`, `typing`
- `integrations.base.BaseIntegration`

## Configuration Variables
None explicitly referenced, assumes integrations load their own.

## Prompts
None.
""",

    "registry.py": """# Documentation Report: registry.py

## Assumptions
- Maintains mapping of `_integrations` (name -> integration) and `_tool_owner` (tool_name -> integration_name).
- `register()` replaces existing tools ownership seamlessly.
- Tool dictionary must declare `"name"`. Tool skipping occurs if name is empty.
- Risk string from tool definition (`"risk": "low" | "medium" | "high" | "read_only"`) determines the safety classification. `"low"` and variations of `"read-only"` map to read-only tools in the AutonomyGovernor, others map to write tools.
- `execute` extracts keyword or positional arguments dynamically based on the tool's execute function signature. Coerces output to `{"success": bool, "data": Any, "error": str}`.
- Exceptions raised in `execute` are caught and wrapped.

## Schema / API Contract
- `register(integration: BaseIntegration)`
- `register_safety_rules(autonomy_governor, risk_evaluator)` scans tools and applies rules.
- `get_tools() -> list[dict[str, Any]]` returns flattened tool schemas.
- `execute(tool_name: str, args: dict[str, Any]) -> dict[str, Any]`

## Dependencies
- `inspect`, `logging`, `typing`
- `integrations.base.BaseIntegration`

## Configuration Variables
None.

## Prompts
None.
""",

    "clients_calendar.py": """# Documentation Report: clients/calendar.py

## Assumptions
- Backed by local `.ics` file at `memory/calendar.ics`
- Mutex lock `_calendar_lock` is used for thread-safe concurrent writes to the file.
- Events are appended manually by splitting on `END:VCALENDAR`.
- `add_event` assumes duration_minutes=60 if not specified.
- `list_events` uses `icalendar` and `dateutil` to parse the file and filter events occurring between now and `days_ahead`.

## Schema / API Contract
- Tool: `add_event(title: str, date: str (YYYY-MM-DD), time: str (HH:MM), duration_minutes: int)` -> `{"event", "date", "time"}`
- Tool: `list_events(days_ahead: int)` -> `{"events": [{"title", "datetime"}]}`

## Dependencies
- `icalendar`, `dateutil` (external)
- `asyncio`, `datetime`, `threading`, `pathlib` (stdlib)

## Configuration Variables
None.

## Prompts
None.
""",

    "clients_computer_control.py": """# Documentation Report: clients/computer_control.py

## Assumptions
- Uses `pyautogui` for all interaction (mouse, keyboard, screenshot).
- Explicitly enables `pyautogui.FAILSAFE` which aborts if mouse is thrown to the corner of the screen.
- Screenshots are forcibly constrained to be inside an `outputs` directory.
- `keyboard_type` defaults to `interval=0.02` per key.

## Schema / API Contract
- Tool: `move_mouse(x: int, y: int)`
- Tool: `mouse_click(x?: int, y?: int, button?: str, double?: bool)`
- Tool: `keyboard_type(text: str, press_enter?: bool, interval?: float)`
- Tool: `take_screenshot(path?: str)` returns absolute path to saved file

## Dependencies
- `pyautogui` (external)
- `asyncio`, `os`, `logging` (stdlib)

## Configuration Variables
None natively, relies on host display server constraints.

## Prompts
None.
""",

    "clients_email.py": """# Documentation Report: clients/email.py

## Assumptions
- Uses standard Python `smtplib` and `imaplib` for SMTP and IMAP operations.
- Assumes `SMTP_PORT` defaults to 587 and supports STARTTLS.
- Searches IMAP by subject using `SUBJECT "query"`.
- Fetching retrieves latest messages and parses headers for From, Subject, Date.

## Schema / API Contract
- Tool: `send_email(to: str, subject: str, body: str)`
- Tool: `read_emails(folder: str, limit: int)`
- Tool: `search_emails(query: str)`

## Dependencies
- `smtplib`, `imaplib`, `email` (stdlib)

## Configuration Variables
- `EMAIL_ADDRESS`
- `EMAIL_PASSWORD`
- `SMTP_HOST`
- `SMTP_PORT` (optional, default 587)
- `IMAP_HOST`

## Prompts
None.
""",

    "clients_github.py": """# Documentation Report: clients/github.py

## Assumptions
- Uses `PyGithub` package.
- Operations map directly to the GitHub API.
- Pull request diff sizes are constrained by `max_files` and `max_patch_chars` to prevent context length explosion. Truncation is explicitly handled and signaled via `truncated` flag.
- Default limit values are usually 20, max limit 100 for pagination/fetching.
- Resolves target repo dynamically from args or falls back to `GITHUB_DEFAULT_REPO`.

## Schema / API Contract
- Tools: `list_open_issues`, `create_issue`, `close_issue`, `list_open_prs`, `get_pr_diff`, `create_gist`, `search_code`.
- `get_pr_diff` returns structured dict containing diff excerpts for each changed file.

## Dependencies
- `github` (PyGithub external)
- `asyncio`, `os`, `typing`

## Configuration Variables
- `GITHUB_TOKEN`
- `GITHUB_DEFAULT_REPO` (optional)

## Prompts
None.
""",

    "clients_gmail.py": """# Documentation Report: clients/gmail.py

## Assumptions
- Uses Gmail API v1 through raw REST calls (`aiohttp`), rather than google-api-python-client.
- Relies on offline `refresh_token` flow to get short-lived `access_token` automatically.
- Email contents are truncated to `2000` chars explicitly to avoid token explosion.
- Unread summary tool uses `task_type="synthesis"` which is likely intercepted by the LLM routing layer.

## Schema / API Contract
- Tools: `list_unread`, `send_gmail`, `summarize_unread`, `mark_as_read`.
- `send_gmail` expects a plain-text email body and builds a base64 encoded MIME object.

## Dependencies
- `aiohttp` (external)
- `base64`, `os`, `email.mime.text` (stdlib)

## Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

## Prompts
None.
""",

    "clients_google_calendar.py": """# Documentation Report: clients/google_calendar.py

## Assumptions
- Uses Google Calendar v3 REST API directly via `aiohttp`.
- Refreshes tokens on-demand via Google OAuth token endpoint.
- Dates are passed natively as ISO-8601 strings, parsed to verify and injected with `timeZone` field in Google request payload.
- `find_free_slot` uses `freeBusy` endpoint to locate gaps between events over `days_ahead`.

## Schema / API Contract
- Tools: `create_event`, `list_events`, `delete_event`, `find_free_slot`.
- `create_event` accepts `summary`, `start`, `end`, `description`, `timezone`, `calendar_id`.

## Dependencies
- `aiohttp` (external)
- `os`, `datetime` (stdlib)

## Configuration Variables
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REFRESH_TOKEN`

## Prompts
None.
""",

    "clients_home_assistant.py": """# Documentation Report: clients/home_assistant.py

## Assumptions
- Interacts with Home Assistant REST API.
- Caches entity states (`/api/states`) for 60 seconds (`_ENTITY_CACHE_TTL_SECONDS`) to avoid spamming HA.
- Distinguishes "sensitive" domains (e.g. `lock`, `alarm_control_panel`) which cannot be modified via convenient toggle endpoints and must go through explicit confirm-gated services.
- Can extract target entity_id, area_id, device_id.

## Schema / API Contract
- Tools: `get_entity_state`, `turn_on_entity`, `turn_off_entity`, `toggle_entity`, `set_thermostat`, `call_service`, `list_entities`.

## Dependencies
- `aiohttp` (external)
- `os`, `time`, `urllib.parse` (stdlib)

## Configuration Variables
- `HOME_ASSISTANT_URL`
- `HOME_ASSISTANT_TOKEN`

## Prompts
None.
""",

    "clients_notion.py": """# Documentation Report: clients/notion.py

## Assumptions
- Integrates with Notion API `v1` version `2022-06-28`.
- Page content appended via `append_block` handles basic validation against allowed types (`paragraph`, `heading_1`, etc.).
- Truncates individual block text insertion to 2000 chars.
- Read operations are marked `low` risk, while page generation and appending block is `confirm`.

## Schema / API Contract
- Tools: `create_page`, `query_database`, `append_block`, `get_page`.
- `create_page` requires a `parent_id` and `title`.

## Dependencies
- `aiohttp` (external)
- `os` (stdlib)

## Configuration Variables
- `NOTION_API_KEY`

## Prompts
None.
""",

    "clients_spotify.py": """# Documentation Report: clients/spotify.py

## Assumptions
- Uses Spotify Web API natively.
- Handles authorization via `refresh_token` and refreshes on *every call*.
- Fails gracefully if there's no "active device" (returns 404).
- Creating playlist requires fetching the current user ID first.
- If playing by query, performs search first and picks the very first result.

## Schema / API Contract
- Tools: `play_track`, `pause`, `search_track`, `get_current_track`, `create_playlist`.

## Dependencies
- `aiohttp` (external)
- `os`, `base64` (stdlib)

## Configuration Variables
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REFRESH_TOKEN`

## Prompts
None.
""",

    "clients_telegram.py": """# Documentation Report: clients/telegram.py

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
""",

    "clients_template.py": """# Documentation Report: clients/template.py

## Assumptions
- Used as boilerplate reference for building new integrations.
- Always returns `False` for availability so it never registers.
- Does not expose tools.

## Schema / API Contract
- Standard `BaseIntegration` schema.

## Dependencies
- `integrations.base.BaseIntegration`

## Configuration Variables
None.

## Prompts
None.
""",

    "clients_weather.py": """# Documentation Report: clients/weather.py

## Assumptions
- Open-Meteo public API (no API key needed).
- Geocodes the city name via `geocoding-api.open-meteo.com` before requesting forecast from `api.open-meteo.com`.
- Extracts `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`.

## Schema / API Contract
- Tool: `get_current_weather(city: str)`.
- Returns dict with `city`, `country`, `temperature_c`, `humidity`, `wind_speed_kmh`.

## Dependencies
- `urllib.request`, `urllib.parse`, `json`, `asyncio`, `logging` (stdlib)

## Configuration Variables
None.

## Prompts
None.
""",

    "clients_whatsapp.py": """# Documentation Report: clients/whatsapp.py

## Assumptions
- Uses Twilio as the backend service to send WhatsApp messages.
- Formats destination to `whatsapp:{to}` and maps sender to `TWILIO_WHATSAPP_FROM`.

## Schema / API Contract
- Tool: `send_whatsapp(to: str, message: str)`.

## Dependencies
- `twilio` (external)
- `os`, `asyncio` (stdlib)

## Configuration Variables
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_WHATSAPP_FROM`

## Prompts
None.
"""
}

for name, content in reports.items():
    safe_name = name.replace("_", "/", 1) if name.startswith("clients_") else name
    filename = safe_name.split("/")[-1]
    
    # Prefix filename properly
    file_path = reports_dir / f"{filename}_Documentation Analyst.md"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Generated {len(reports)} markdown reports.")
