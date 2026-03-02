# JARVIS — FUTURE BLUEPRINT
**Status:** Engineering Spec for All Future Sessions  
**Author:** Senior Engineering Audit  
**Scope:** Every new capability, API integration, and architectural upgrade needed to make Jarvis genuinely useful  
**How to use:** Paste relevant section at the start of each new LLM session. One section = one session. Do not try to do two sections at once.

---

## HOW TO READ THIS FILE

Each section has:
- **Goal** — what you're building and why
- **Do** — exact implementation instructions
- **Don't** — explicit anti-patterns the LLM must not do
- **Files to create** — exact paths
- **Files to modify** — exact paths with what changes
- **Verification** — commands to run before marking complete

All new integrations follow the `BaseIntegration` pattern. All new tools go through the risk evaluator. No exceptions.

---

## SESSION PRIORITY ORDER

```
IMMEDIATE  → Cloud LLM Fallback (Jarvis is dead when Ollama is offline)
IMMEDIATE  → Telegram Notifications (cheapest, most useful comms integration)
SHORT      → Google Calendar API (replace the broken ICS parser)
SHORT      → Notion Integration (personal knowledge base)
SHORT      → Todoist/Task Management
SHORT      → Spotify Control
MEDIUM     → Home Assistant
MEDIUM     → GitHub Integration
MEDIUM     → Memory Consolidation Engine
MEDIUM     → Daily Briefing Generator
LONG       → Sub-agent spawning
LONG       → Self-improvement loop
LONG       → ElevenLabs TTS upgrade
LONG       → Codebase cleanup (controller/, logging/ bloat)
```

---

## SESSION A — CLOUD LLM FALLBACK

### Goal
When Ollama is offline (laptop battery, no GPU, remote use), Jarvis must not go silent. It must fall back to a cloud provider automatically, transparently, with the user's consent configured once in settings.env.

### New Files
```
core/llm/cloud_client.py     — abstract cloud client with Groq, OpenAI, Anthropic adapters
core/llm/llm_router.py       — unified router: tries Ollama first, then cloud chain
```

### `core/llm/cloud_client.py`
```python
from __future__ import annotations
import os, logging
from typing import Any
logger = logging.getLogger(__name__)

class CloudLLMClient:
    """
    Unified interface for cloud LLM providers.
    Priority: Groq (fastest, cheapest) → OpenAI → Anthropic → fail loudly.
    """
    
    PROVIDERS = ["groq", "openai", "anthropic"]
    
    def __init__(self) -> None:
        self._available: list[str] = []
        for provider in self.PROVIDERS:
            if self._check_provider(provider):
                self._available.append(provider)
        if not self._available:
            logger.warning("No cloud LLM providers configured. Cloud fallback disabled.")
    
    def _check_provider(self, name: str) -> bool:
        keys = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        return bool(os.environ.get(keys.get(name, "")))
    
    async def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        for provider in self._available:
            try:
                result = await self._call(provider, prompt, system, temperature)
                if result:
                    logger.info("Cloud LLM response from '%s'", provider)
                    return result
            except Exception as exc:
                logger.warning("Cloud provider '%s' failed: %s", provider, exc)
        raise RuntimeError("All cloud LLM providers failed or are unconfigured.")
    
    async def _call(self, provider: str, prompt: str, system: str, temperature: float) -> str:
        if provider == "groq":
            return await self._call_groq(prompt, system, temperature)
        if provider == "openai":
            return await self._call_openai(prompt, system, temperature)
        if provider == "anthropic":
            return await self._call_anthropic(prompt, system, temperature)
        return ""
    
    async def _call_groq(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp, os
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",  # fastest Groq model as of 2025
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2048,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    
    async def _call_openai(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp, os
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",  # cheapest capable model
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    
    async def _call_anthropic(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp, os
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 2048,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                return data["content"][0]["text"]
```

### Modify `core/llm/client.py`
```python
# In LLMClientV2.complete() — add cloud fallback at the end of the method

async def complete(self, prompt: str, system: str = "", temperature: float = 0.1, ...) -> str:
    # ... existing Ollama logic ...
    
    # If Ollama returned empty, try cloud fallback
    if not raw and self._cloud_client is not None:
        logger.warning("Ollama returned empty. Attempting cloud fallback.")
        try:
            return await self._cloud_client.complete(prompt, system=system, temperature=temperature)
        except Exception as exc:
            logger.error("Cloud fallback also failed: %s", exc)
    return raw or ""
```

### settings.env additions
```ini
# Cloud LLM Fallback — leave empty to disable
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
CLOUD_LLM_FALLBACK_ENABLED=true
```

### Do
- ✅ Always try Ollama first. Cloud is fallback only.
- ✅ Log clearly which provider responded so the user knows when they're using cloud.
- ✅ Respect `CLOUD_LLM_FALLBACK_ENABLED=false` to fully disable cloud.
- ✅ Handle API key missing gracefully — skip that provider, don't crash.

### Don't
- ❌ Don't send the full workspace file map to cloud providers — it's expensive and a privacy leak.
- ❌ Don't stream from cloud providers in the same path as Ollama streaming.
- ❌ Don't hardcode model names in the client — put them in `config/jarvis.ini [cloud_models]`.

### Verification
```bash
GROQ_API_KEY=your_key python -c "
import asyncio
from core.llm.cloud_client import CloudLLMClient
c = CloudLLMClient()
result = asyncio.run(c.complete('Say hello in one word.'))
assert result, 'FAIL: no response'
print('PASS:', result)
"
```

---

## SESSION B — TELEGRAM NOTIFICATIONS

### Goal
Jarvis needs a reliable push channel. Email is too slow. Twilio costs money. Telegram is free, instant, has a bot API, works on every platform, and the library is mature. This is the most impactful one-session integration.

### New Files
```
integrations/clients/telegram.py
```

### settings.env additions
```ini
TELEGRAM_BOT_TOKEN=         # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID=           # Your personal chat ID (send /start to your bot, then GET /getUpdates)
```

### `integrations/clients/telegram.py`
```python
from __future__ import annotations
import asyncio, os, logging
from typing import Any
import aiohttp
from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

class TelegramIntegration(BaseIntegration):
    name = "telegram"
    description = "Send messages, files, and alerts to Telegram"
    required_config = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

    def is_available(self) -> bool:
        return all(bool(os.environ.get(k)) for k in self.required_config)

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "send_telegram_message",
                "description": "Send a text message to the user's Telegram",
                "risk": "low",  # It's YOUR bot messaging YOU — low risk
                "args": {
                    "message": {"type": "string", "description": "Message text (supports Markdown)"},
                    "parse_mode": {"type": "string", "default": "Markdown"},
                },
                "required_args": ["message"],
            },
            {
                "name": "send_telegram_alert",
                "description": "Send an urgent alert with high priority notification",
                "risk": "low",
                "args": {
                    "message": {"type": "string", "description": "Alert text"},
                },
                "required_args": ["message"],
            },
        ]

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        try:
            if tool_name == "send_telegram_message":
                data = await self._send(str(args.get("message", "")), parse_mode=str(args.get("parse_mode", "Markdown")))
                return {"success": True, "data": data, "error": None}
            if tool_name == "send_telegram_alert":
                data = await self._send(f"🚨 *ALERT*\n{args.get('message', '')}", disable_notification=False)
                return {"success": True, "data": data, "error": None}
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            logger.error("Telegram execute failed: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}

    async def _send(self, text: str, parse_mode: str = "Markdown", disable_notification: bool = False) -> dict:
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]
        url = TELEGRAM_API.format(token=token, method="sendMessage")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise RuntimeError(f"Telegram API error: {data.get('description')}")
                return {"message_id": data["result"]["message_id"]}
```

### Register in risk_evaluator.py
```python
# Add to _DEFAULT_LOW set:
"send_telegram_message", "send_telegram_alert",
```

### Do
- ✅ Use Markdown formatting so Jarvis responses look clean on phone.
- ✅ Send proactive alerts (goals completed, system warnings) through this channel.
- ✅ Wire the `NotificationManager` in `core/proactive/notifier.py` to use Telegram as primary channel.

### Don't
- ❌ Don't send full conversation history to Telegram — it's a notification channel, not a chat mirror.
- ❌ Don't implement incoming message polling (webhooks) in this session — receive is out of scope.

### Verification
```bash
python -c "
import asyncio, os
os.environ['TELEGRAM_BOT_TOKEN'] = 'your_token'
os.environ['TELEGRAM_CHAT_ID'] = 'your_chat_id'
from integrations.clients.telegram import TelegramIntegration
t = TelegramIntegration()
assert t.is_available()
result = asyncio.run(t.execute('send_telegram_message', {'message': 'Jarvis test ✅'}))
assert result['success'], result['error']
print('PASS')
"
```

---

## SESSION C — GOOGLE CALENDAR API

### Goal
Replace the hand-rolled ICS regex parser with a proper Google Calendar integration. The ICS file approach is dead the moment someone creates a recurring event or uses timezones. Google Calendar is where everyone's actual schedule lives.

### New Files
```
integrations/clients/google_calendar.py
```

### Installation
```bash
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
```

### settings.env additions
```ini
GOOGLE_CREDENTIALS_FILE=config/google_credentials.json   # Download from Google Cloud Console
GOOGLE_TOKEN_FILE=config/google_token.json               # Auto-created on first auth
```

### `integrations/clients/google_calendar.py`
```python
from __future__ import annotations
import asyncio, os, logging
from datetime import datetime, timedelta, timezone
from typing import Any
from integrations.base import BaseIntegration

logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/calendar"]

class GoogleCalendarIntegration(BaseIntegration):
    name = "google_calendar"
    description = "Read and write Google Calendar events"
    required_config = ["GOOGLE_CREDENTIALS_FILE"]

    def is_available(self) -> bool:
        try:
            import googleapiclient  # noqa: F401
            return os.path.exists(os.environ.get("GOOGLE_CREDENTIALS_FILE", ""))
        except ImportError:
            return False

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_calendar_events",
                "description": "List upcoming events from Google Calendar",
                "risk": "low",
                "args": {"days_ahead": {"type": "integer", "default": 7}},
                "required_args": [],
            },
            {
                "name": "create_calendar_event",
                "description": "Create a new event in Google Calendar",
                "risk": "confirm",
                "args": {
                    "title": {"type": "string"},
                    "start_datetime": {"type": "string", "description": "ISO 8601: 2024-12-25T14:00:00"},
                    "end_datetime": {"type": "string"},
                    "description": {"type": "string", "default": ""},
                    "attendees": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required_args": ["title", "start_datetime", "end_datetime"],
            },
            {
                "name": "find_free_slots",
                "description": "Find free time slots in the next N days",
                "risk": "low",
                "args": {
                    "days_ahead": {"type": "integer", "default": 3},
                    "duration_minutes": {"type": "integer", "default": 60},
                },
                "required_args": [],
            },
        ]

    def _get_service(self):
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        import pickle

        creds = None
        token_file = os.environ.get("GOOGLE_TOKEN_FILE", "config/google_token.json")
        
        if os.path.exists(token_file):
            with open(token_file, "rb") as f:
                creds = pickle.load(f)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    os.environ["GOOGLE_CREDENTIALS_FILE"], SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(token_file, "wb") as f:
                pickle.dump(creds, f)
        
        return build("calendar", "v3", credentials=creds)

    async def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = args or {}
        loop = asyncio.get_running_loop()
        try:
            if tool_name == "list_calendar_events":
                data = await loop.run_in_executor(None, lambda: self._list_events(int(args.get("days_ahead", 7))))
                return {"success": True, "data": data, "error": None}
            if tool_name == "create_calendar_event":
                data = await loop.run_in_executor(None, lambda: self._create_event(args))
                return {"success": True, "data": data, "error": None}
            if tool_name == "find_free_slots":
                data = await loop.run_in_executor(None, lambda: self._find_free(
                    int(args.get("days_ahead", 3)),
                    int(args.get("duration_minutes", 60))
                ))
                return {"success": True, "data": data, "error": None}
            return {"success": False, "data": None, "error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            logger.error("Google Calendar error: %s", exc)
            return {"success": False, "data": None, "error": str(exc)}

    def _list_events(self, days_ahead: int) -> dict:
        service = self._get_service()
        now = datetime.now(timezone.utc).isoformat()
        end = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()
        result = service.events().list(
            calendarId="primary", timeMin=now, timeMax=end,
            maxResults=20, singleEvents=True, orderBy="startTime"
        ).execute()
        events = []
        for e in result.get("items", []):
            start = e["start"].get("dateTime", e["start"].get("date"))
            events.append({"title": e.get("summary", ""), "start": start, "id": e["id"]})
        return {"events": events}

    def _create_event(self, args: dict) -> dict:
        service = self._get_service()
        event = {
            "summary": args["title"],
            "description": args.get("description", ""),
            "start": {"dateTime": args["start_datetime"], "timeZone": "UTC"},
            "end": {"dateTime": args["end_datetime"], "timeZone": "UTC"},
        }
        if args.get("attendees"):
            event["attendees"] = [{"email": e} for e in args["attendees"]]
        result = service.events().insert(calendarId="primary", body=event).execute()
        return {"event_id": result["id"], "link": result.get("htmlLink")}

    def _find_free(self, days_ahead: int, duration_minutes: int) -> dict:
        service = self._get_service()
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days_ahead)
        body = {
            "timeMin": now.isoformat(),
            "timeMax": end.isoformat(),
            "items": [{"id": "primary"}],
        }
        result = service.freebusy().query(body=body).execute()
        busy = result.get("calendars", {}).get("primary", {}).get("busy", [])
        # Return first available slot — full algorithm left as exercise
        return {"busy_blocks": busy, "duration_requested_minutes": duration_minutes}
```

### Do
- ✅ Use OAuth2 — never use a service account for personal calendar access.
- ✅ Store the token with pickle so OAuth only runs once.
- ✅ Handle token refresh automatically.

### Don't
- ❌ Don't delete calendar events — only create, read, and update. Deletion is irreversible.
- ❌ Don't hardcode `calendarId` — always use `"primary"` which resolves to the user's main calendar.
- ❌ Don't run the OAuth browser flow in an async context — use `run_in_executor`.

---

## SESSION D — NOTION INTEGRATION

### Goal
Notion is where many people keep their notes, tasks, and knowledge base. If Jarvis can read and write Notion, it becomes a genuine personal assistant that knows your life context.

### New Files
```
integrations/clients/notion.py
```

### Installation
```bash
pip install notion-client
```

### settings.env additions
```ini
NOTION_API_KEY=                    # From notion.so/my-integrations
NOTION_TASKS_DATABASE_ID=          # The Notion database to read/write tasks
NOTION_NOTES_PAGE_ID=              # Optional: a notes page for Jarvis to write to
```

### `integrations/clients/notion.py` — Tools to implement
```
get_tasks          — Query tasks database, filter by status/due date
create_task        — Create new page in tasks database with title + due date
complete_task      — Update a task page's status property to "Done"
search_notes       — Full-text search across all accessible pages
append_to_page     — Append a paragraph block to a specific page
create_note_page   — Create a new page with title and content
```

### Do
- ✅ Use the official `notion-client` SDK, not raw HTTP.
- ✅ Map Notion database properties by type (title, date, select, checkbox) — don't assume string.
- ✅ Cache the database schema (property names) on first call so you don't re-fetch every time.

### Don't
- ❌ Don't archive or delete Notion pages. Ever. Notion trash is permanent after 30 days.
- ❌ Don't read pages the user has not shared with the integration — Notion's API enforces this but be explicit.
- ❌ Don't assume property names. Query the database schema first and map dynamically.

---

## SESSION E — SPOTIFY CONTROL

### Goal
Voice command music control. "Play lo-fi beats", "skip this song", "turn up the volume". This is a quality-of-life feature that makes Jarvis feel like a real assistant.

### New Files
```
integrations/clients/spotify.py
```

### Installation
```bash
pip install spotipy
```

### settings.env additions
```ini
SPOTIFY_CLIENT_ID=           # From developer.spotify.com
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
SPOTIFY_DEVICE_ID=           # Optional: target specific device
```

### Tools to implement
```
play_track           — Search + play by name ("play Daft Punk")
play_playlist        — Play a named playlist
pause_playback       — Pause current track
resume_playback      — Resume
skip_track           — Skip to next
previous_track       — Go back
set_volume           — 0–100
get_current_track    — Return track name + artist + album
search_tracks        — Return list of matches for a query
```

### Do
- ✅ Use spotipy's OAuth cache so auth only runs once.
- ✅ Handle "no active device" gracefully — tell the user to open Spotify first.
- ✅ `play_track` should search first, then play the top result, then confirm to user.

### Don't
- ❌ Don't use `premium_required` features if the user has a free account — check scope.
- ❌ Don't spam the API — Spotify rate limits aggressively (10 req/sec, 180 req/minute).
- ❌ Don't hardcode playlist IDs — search by name so it works for any user.

---

## SESSION F — HOME ASSISTANT INTEGRATION

### Goal
Control smart home devices through Jarvis voice commands. "Turn off the living room lights", "set thermostat to 22 degrees", "lock the front door".

### New Files
```
integrations/clients/home_assistant.py
```

### settings.env additions
```ini
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=              # Long-lived access token from HA profile
```

### Tools to implement
```
get_entity_state       — "Is the front door locked?"
turn_on_entity         — "Turn on kitchen lights"
turn_off_entity        — "Turn off all lights"
toggle_entity          — Toggle any switch/light/fan
set_thermostat         — Set temperature target
call_service           — Generic service call for anything not covered above
list_entities          — Return all entities by domain (light, switch, sensor...)
```

### Do
- ✅ Use the HA REST API, not WebSocket — simpler and sufficient for this use case.
- ✅ All destructive calls (lock, alarm, door) must have `"risk": "confirm"`.
- ✅ Cache the entity list for 60 seconds — it doesn't change often.

### Don't
- ❌ Don't call `lock` or `alarm_control_panel` services without double-confirm in the risk evaluator.
- ❌ Don't allow Jarvis to autonomously run automations — only respond to explicit user commands.

---

## SESSION G — GITHUB INTEGRATION

### Goal
Jarvis becomes a developer assistant. "Create a GitHub issue for this bug", "show me open PRs", "commit this file to my repo". Directly useful for developers using Jarvis as a coding assistant.

### New Files
```
integrations/clients/github.py
```

### Installation
```bash
pip install PyGithub
```

### settings.env additions
```ini
GITHUB_TOKEN=                    # Personal access token with repo scope
GITHUB_DEFAULT_REPO=             # e.g. "username/reponame"
```

### Tools to implement
```
list_open_issues     — Filter by label, assignee, milestone
create_issue         — Title + body + labels
close_issue          — By issue number
list_open_prs        — Show open pull requests with status
get_pr_diff          — Return diff of a PR (for code review)
create_gist          — Save a code snippet as a gist
search_code          — Search within a repo by keyword
```

### Do
- ✅ Default to `GITHUB_DEFAULT_REPO` but allow overriding per-call with `repo=` arg.
- ✅ `create_issue` and `close_issue` are `risk: confirm`.
- ✅ Use pagination — never assume all results fit in first page.

### Don't
- ❌ Don't allow push, force push, or branch deletion through Jarvis — too destructive.
- ❌ Don't include the full PR diff in a chat response — summarize with the LLM first.

---

## SESSION H — MEMORY CONSOLIDATION ENGINE

### Goal
Right now Jarvis memory grows forever. After enough sessions, the SQLite DB is bloated with redundant facts and stale observations. This session builds a nightly job that compresses episodic memories into semantic summaries, then prunes the raw episodes.

### New Files
```
core/memory/consolidation.py
core/memory/consolidation_scheduler.py
```

### `core/memory/consolidation.py` — What to build
```python
class MemoryConsolidator:
    """
    Runs nightly (or on-demand) to compress old episodic memories.
    
    Algorithm:
    1. Fetch all episodic memories older than 7 days
    2. Group by topic using embedding clustering (cosine similarity > 0.85)
    3. For each cluster, call LLM: "Summarize these memories into 2-3 key facts"
    4. Write the summary as a new semantic memory with source="consolidation"
    5. Delete the original episodic entries
    6. Log: how many entries pruned, how many summaries created
    """
    
    async def run(self, memory: HybridMemory, llm: LLMClientV2) -> dict:
        ...
    
    async def consolidate_cluster(self, entries: list[dict], llm: LLMClientV2) -> str:
        ...
```

### Do
- ✅ Run as a background task — never block the main agent loop.
- ✅ Keep a `consolidation_log` table in SQLite — what was pruned, when, summary quality score.
- ✅ Never consolidate memories from the last 7 days — they're still hot context.
- ✅ Test with a mock LLM to avoid burning API credits during development.

### Don't
- ❌ Don't delete episodic memories until the summary is successfully written and verified.
- ❌ Don't run consolidation during active voice sessions — wait for idle.
- ❌ Don't use consolidation on user-pinned or flagged memories.

---

## SESSION I — DAILY BRIEFING GENERATOR

### Goal
Every morning, Jarvis compiles a structured briefing: today's calendar events, weather, active goals, unread priority emails, news headlines. Delivered via Telegram (Session B) and optionally spoken via TTS.

### New Files
```
core/proactive/daily_briefing.py
```

### What the briefing contains
```
1. Date + greeting ("Good morning, it's Monday")
2. Weather for the day
3. Today's calendar events (from Google Calendar)
4. Active goals and their status
5. Top 3 news headlines (from NewsAPI or RSS)
6. Any overdue tasks
7. System health (Ollama status, memory size)
```

### `core/proactive/daily_briefing.py`
```python
class DailyBriefingGenerator:
    """
    Runs at a scheduled time (default 8:00 AM) and generates a briefing.
    Sends via Telegram integration. Optionally speaks via TTS.
    """
    
    async def generate(self, controller) -> str:
        """Fetch all data sources, synthesize with LLM, return formatted text."""
        ...
    
    async def _fetch_weather(self) -> str:
        ...
    
    async def _fetch_calendar(self) -> str:
        ...
    
    async def _fetch_goals(self, goal_manager) -> str:
        ...
    
    async def synthesize(self, sections: list[str], llm) -> str:
        """Pass all sections to LLM with: 'Write a concise morning briefing.'"""
        ...
```

### Do
- ✅ Fetch all data concurrently with `asyncio.gather()`.
- ✅ If any data source fails, skip it gracefully — don't cancel the whole briefing.
- ✅ Keep the briefing under 400 words — it's a briefing, not an essay.
- ✅ Schedule with `apscheduler` in `core/agentic/scheduler.py`.

### Don't
- ❌ Don't include sensitive email content in briefing unless user explicitly opts in.
- ❌ Don't generate the briefing if Jarvis was not running at the scheduled time — skip to next day.

---

## SESSION J — ARCHITECTURAL CLEANUP

### Goal
Purge the dead weight. This session is pure cleanup — no new features. The codebase has hundreds of files that don't belong here and make every LLM session slower and more confused.

### Files to delete or move
```bash
# Move to archive_legacy/ — never delete, just quarantine
mkdir -p archive_legacy/{llm_bloat,logging_bloat,controller_bloat,voice_bloat}

# core/llm/ — third-party schema files
mv core/llm/fastjsonschema_*.py archive_legacy/llm_bloat/
mv core/llm/json_schema_test_suite.py archive_legacy/llm_bloat/
mv core/llm/test_*.py archive_legacy/llm_bloat/
mv core/llm/useless_applicator_schemas.py archive_legacy/llm_bloat/
mv core/llm/schemapi.py archive_legacy/llm_bloat/
mv core/llm/_generate_schema.py core/llm/_schema_gather.py core/llm/_schema_generation_shared.py archive_legacy/llm_bloat/
mv core/llm/configuration_smollm3.py core/llm/modeling_smollm3.py core/llm/modular_smollm3.py archive_legacy/llm_bloat/

# core/logging/ — PyTorch/gRPC internals
mv core/logging/_logsumexp.py core/logging/_logistic.py archive_legacy/logging_bloat/
mv core/logging/bench_discrete_log.py core/logging/c10d_logger.py archive_legacy/logging_bloat/
mv core/logging/logging_tensor.py archive_legacy/logging_bloat/
mv core/logging/Logo_pb2.py core/logging/logs_pb2.py core/logging/logs_pb2_grpc.py archive_legacy/logging_bloat/
mv core/logging/logging_pb2.py archive_legacy/logging_bloat/
mv core/logging/prolog.py core/logging/morphology.py core/logging/_morphology.py archive_legacy/logging_bloat/

# core/controller/ — Kubernetes + PyTorch state files
mv core/controller/v1_*.py archive_legacy/controller_bloat/
mv core/controller/ClientState_pb2.py core/controller/WidgetStates_pb2.py archive_legacy/controller_bloat/
mv core/controller/_fsdp_state.py core/controller/_pybind_state.py archive_legacy/controller_bloat/
mv core/controller/_backward_state.py core/controller/_trace_state.py archive_legacy/controller_bloat/

# core/voice/ — HuggingFace models not used by Jarvis voice pipeline
# Keep: audio.py, audio_input.py, audio_playback.py, stt.py, tts.py, voice.py, voice_layer.py, voice_loop.py, wake_word.py
# Move everything else:
mv core/voice/configuration_*.py archive_legacy/voice_bloat/
mv core/voice/modeling_*.py archive_legacy/voice_bloat/
mv core/voice/modular_*.py archive_legacy/voice_bloat/
mv core/voice/processing_*.py archive_legacy/voice_bloat/
mv core/voice/image_processing_*.py archive_legacy/voice_bloat/
mv core/voice/feature_extraction_*.py archive_legacy/voice_bloat/
mv core/voice/dummy_*.py archive_legacy/voice_bloat/
```

### After cleanup, run full verification
```bash
pytest tests/ -q --timeout=30
python main.py --verify
python -c "from core.controller_v2 import JarvisControllerV2; print('controller OK')"
python -c "from core.llm.client import LLMClientV2; print('llm client OK')"
python -c "from integrations.loader import load_all; print('integrations OK')"
```

### Do
- ✅ Move, never delete — in case something imports from these files.
- ✅ Run the full test suite after each batch of moves.
- ✅ Add `norecursedirs` to pytest.ini after moving so test runner ignores archive dirs.

### Don't
- ❌ Don't move files without running the test suite after each batch.
- ❌ Don't touch `core/vision/` in this session — it has live dependencies.
- ❌ Don't delete the archive dirs themselves — future audit trails matter.

---

## LLM DO'S AND DON'TS — PASTE AT EVERY SESSION START

```
ARCHITECTURE RULES (non-negotiable):
DO:  Every new integration = BaseIntegration subclass in integrations/clients/
DO:  Every new tool = registered in risk_evaluator with a risk level
DO:  All paths = Path(__file__).parent / "..." — never absolute
DO:  All I/O inside integration execute() = async or run_in_executor()
DO:  All secrets = os.environ.get() from settings.env — never hardcoded
DO:  All irreversible tools = risk: "confirm" — user must approve
DO:  One test file per integration: tests/test_{name}.py minimum
DON'T: import from archive_legacy/, archive_jarvis_duplicate/, or Failed/
DON'T: call asyncio.new_event_loop() or .run_until_complete() from async context
DON'T: import LLMClientV2 from core.llm.llm_v2 — use core.llm.client
DON'T: add files to core/controller/ unless they manage agent/session state
DON'T: create new entry point files (main_v4.py etc.) — use main.py subcommands
DON'T: commit *.env, *.exe, outputs/, logs/, *.jsonl, __pycache__/ to git
DON'T: skip the ModelRouter — all LLM calls go through LLMClientV2.complete()
DON'T: inject full workspace file tree into every prompt — only when query mentions files
DON'T: let agent loop take irreversible action without risk_evaluator approval
DON'T: create a new Python file if an existing file already handles that concern
```

---

## QUICK REFERENCE — ALL NEW ENV VARS BY SESSION

| Session | Variable | Source |
|---------|----------|--------|
| A | `GROQ_API_KEY` | console.groq.com |
| A | `OPENAI_API_KEY` | platform.openai.com |
| A | `ANTHROPIC_API_KEY` | console.anthropic.com |
| A | `CLOUD_LLM_FALLBACK_ENABLED` | true/false |
| B | `TELEGRAM_BOT_TOKEN` | @BotFather on Telegram |
| B | `TELEGRAM_CHAT_ID` | /getUpdates after /start |
| C | `GOOGLE_CREDENTIALS_FILE` | Google Cloud Console |
| C | `GOOGLE_TOKEN_FILE` | Auto-created |
| D | `NOTION_API_KEY` | notion.so/my-integrations |
| D | `NOTION_TASKS_DATABASE_ID` | From database URL |
| E | `SPOTIFY_CLIENT_ID` | developer.spotify.com |
| E | `SPOTIFY_CLIENT_SECRET` | developer.spotify.com |
| E | `SPOTIFY_REDIRECT_URI` | http://localhost:8888/callback |
| F | `HOME_ASSISTANT_URL` | http://homeassistant.local:8123 |
| F | `HOME_ASSISTANT_TOKEN` | HA profile → Long-lived tokens |
| G | `GITHUB_TOKEN` | github.com → Settings → Developer settings |
| G | `GITHUB_DEFAULT_REPO` | username/reponame |

---

## METRIC — WHAT "DONE" LOOKS LIKE FOR EACH SESSION

A session is only complete when ALL of these pass:

```bash
# 1. Integration loads without error
python -c "from integrations.loader import IntegrationLoader; print('loader OK')"

# 2. New integration is available
python -c "
from integrations.clients.{name} import {ClassName}
i = {ClassName}()
print('available:', i.is_available())
print('tools:', [t['name'] for t in i.get_tools()])
"

# 3. Test file passes
pytest tests/test_{name}.py -v

# 4. Full suite still passes
pytest tests/ -q --timeout=30

# 5. Risk evaluator knows about new tools
python -c "
from core.autonomy.risk_evaluator import RiskEvaluator
r = RiskEvaluator()
# verify new tool names are in the evaluator tables
"

# 6. Main still starts
python main.py --verify
```

---

*Sessions A and B are mandatory before any other new feature. The system has no cloud fallback and no reliable notification channel — those are foundational gaps. Fix JARVIS_FIXES.md first, then do sessions A and B, then continue in order.*
