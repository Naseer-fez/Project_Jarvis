# Jarvis

Jarvis is a local-first personal AI runtime built around persistent memory, model routing, optional voice interaction, a FastAPI dashboard, risk-gated tool execution, and dynamically loaded integrations.

This repository has legacy modules from earlier phases, but the current runtime entrypoint is `main.py` and the active controller is `core/controller_v2.py`.

## What the system does

- Runs as a local CLI assistant, optional voice assistant, dashboard-backed service, or headless daemon
- Persists preferences, conversations, and episodes in SQLite, with optional semantic recall through ChromaDB
- Routes LLM tasks across local Ollama models first, with optional cloud fallback if configured
- Supports proactive goals, reminders, scheduling, and background resource monitoring
- Loads integrations at runtime based on installed packages and available credentials
- Enforces execution safety through risk tiers, confirmation gates, sandboxed file boundaries, and audit logging
- Produces operational artifacts such as app logs, audit logs, execution traces, and workflow logs

## Core capabilities

### Runtime modes

- Text CLI
- Voice mode
- Web dashboard
- Headless service mode
- Health check and audit verification modes

### Intelligence and memory

- Hybrid memory with SQLite plus semantic retrieval
- User profile extraction and synthesis
- Goal tracking and scheduled reminders
- Local model routing by task type: planning, chat, vision, synthesis, fallback
- Optional cloud LLM fallback chain: Gemini, Groq, OpenAI, Anthropic

### Safety and operations

- Risk evaluator with blocked, high-risk, medium-risk, and low-risk actions
- Confirmation gating for write and outbound actions
- Safe-directory execution boundaries
- GUI automation disabled by default
- Hash-chained audit log verification
- Startup health diagnostics with non-zero exit on failures

### Integrations

- Local calendar
- Weather
- Telegram
- Gmail
- Google Calendar
- Notion
- Spotify
- IMAP/SMTP email
- Twilio WhatsApp

## Architecture

```text
User / Voice / Dashboard
        |
        v
     main.py
        |
        v
JarvisControllerV2
  |- HybridMemory (SQLite + Chroma)
  |- LLMClientV2 + ModelRouter
  |- UserProfileEngine + ProfileSynthesizer
  |- GoalManager + Scheduler
  |- BackgroundMonitor + NotificationManager
  |- Optional VoiceLayer
  |- Dynamic IntegrationRegistry
  |- WorkflowEngine / Dispatcher / Risk controls
        |
        v
Tools, Integrations, Logs, Audit Artifacts
```

## Repository layout

```text
.
|- main.py                     # Primary runtime entrypoint
|- Main_No_Voice.py            # Text-only compatibility entrypoint
|- requirements.txt
|- .env.example
|- config/
|  |- jarvis.ini               # Primary runtime configuration
|  `- settings.env.template    # Additional optional secrets and provider vars
|- core/
|  |- controller_v2.py         # Current main controller
|  |- llm/                     # LLM clients, routing, cloud fallback
|  |- memory/                  # SQLite, Chroma, compression, compatibility shims
|  |- voice/                   # Wake word, STT, TTS, voice loop
|  |- execution/               # Tool dispatch, sandboxing, rollback support
|  |- workflow/                # DAG-style workflow engine
|  |- autonomy/                # Risk and autonomy controls
|  |- agentic/                 # Goals, scheduler, reflection, context
|  `- introspection/           # Health and explainability helpers
|- integrations/
|  |- clients/                 # Runtime-loaded integrations
|  |- loader.py                # Discovery and bootstrap
|  `- registry.py              # Tool registry and execution router
|- dashboard/
|  |- server.py                # FastAPI dashboard
|  |- templates/
|  `- static/
|- data/                       # Runtime data, SQLite DB, Chroma store
|- logs/                       # App log and audit log
|- outputs/                    # Session traces and execution artifacts
|- runtime/                    # Runtime control files
`- tests/                      # Automated test suite
```

## Prerequisites

### Required

- Python 3.10 or newer
- A virtual environment
- Ollama for local-first operation

### Recommended local models

Pull the models that match the default routing in `config/jarvis.ini`:

```powershell
ollama serve
ollama pull deepseek-r1:8b
ollama pull mistral:7b
ollama pull llava
```

### Optional packages for extended features

`requirements.txt` installs the main runtime, but several optional features depend on additional packages:

```powershell
pip install psutil icalendar python-dateutil twilio google-genai pvporcupine colorama
```

Install only what you need:

- `psutil` for proactive resource monitoring
- `icalendar` and `python-dateutil` for richer local calendar support
- `twilio` for WhatsApp integration
- `google-genai` for Gemini SDK-backed execution paths
- `pvporcupine` for wake-word support
- `colorama` for improved health-check terminal output

## Installation

### 1. Clone and create a virtual environment

```powershell
git clone <your-repo-url> Jarvis
cd Jarvis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment secrets

Use `.env.example` as the base file:

```powershell
Copy-Item .env.example .env
```

If you plan to use email, WhatsApp, extra cloud providers, or additional voice settings, also review:

```text
config/settings.env.template
```

### 3. Review the main configuration

The primary runtime config lives in:

```text
config/jarvis.ini
```

Important defaults:

- Version: `2.0.0`
- Memory database: `data/jarvis_memory.db`
- Chroma directory: `data/chroma`
- App log: `logs/app.log`
- Audit log: `logs/audit.jsonl`
- Dashboard default host: `127.0.0.1`
- Dashboard default port: `7070`
- Safe execution directories: `workspace`, `outputs`, `data`

### 4. Verify the installation

```powershell
python main.py --health-check
```

Expected behavior:

- Exit code `0` if startup checks pass
- Non-zero exit if a required subsystem fails, such as Ollama being unreachable

## Running Jarvis

### Standard CLI

```powershell
python main.py
```

### Voice mode

```powershell
python main.py --voice
```

### Dashboard plus runtime

```powershell
python main.py --gui
```

This starts the runtime and serves the dashboard at `http://127.0.0.1:7070` by default.

### Headless service with dashboard

```powershell
python main.py --headless --gui
```

### Health diagnostics

```powershell
python main.py --health-check
python main.py --health-check --strict-health
```

### Audit verification

```powershell
python main.py --verify
```

### Model inspection

```powershell
python main.py --list-models
python main.py --print-config
```

### Backward-compatible text-only entrypoint

```powershell
python Main_No_Voice.py
```

## Runtime behavior

### CLI examples

Jarvis accepts both explicit commands and natural-language inputs:

- `help`
- `status`
- `exit`
- `remember i like coffee`
- `my name is Fez`
- `i prefer concise answers`
- `remind me to review the deployment checklist tomorrow`
- `show goals`

### Goal and reminder behavior

The current controller recognizes phrases such as:

- `remind me to ...`
- `set goal ...`
- `remember to ...`
- `don't forget to ...`
- `schedule ...`

Goals are tracked through the goal manager and checked by a background scheduler.

## Configuration reference

### Primary runtime config: `config/jarvis.ini`

| Section | Purpose |
| --- | --- |
| `general` | System identity, version, environment |
| `ollama` | Local Ollama endpoint, planner model, vision model, timeouts |
| `models` | Ordered model candidates for planning, chat, vision, synthesis, fallback |
| `memory` | SQLite path, Chroma path, embedding model, recall limits, decay behavior |
| `risk` | Forbidden actions, blocked actions, confirmation rules, failsafe behavior |
| `execution` | Safe directories, read limits, app allowlist, GUI automation, rollback, worker counts |
| `hardware` | Serial device settings |
| `logging` | App log path, audit log path, trace directory, log level |
| `voice` | Wake word, STT, TTS, silence thresholds, audio settings |
| `concurrency` | Parallel task limit |
| `plugins` | Plugin directory and scope selection |
| `proactive` | CPU and RAM alert thresholds, goal check interval |
| `dashboard` | Runtime control file and dashboard binding inputs |

### Core environment variables

| Variable | Purpose |
| --- | --- |
| `JARVIS_ENV` | Runtime environment name |
| `JARVIS_CONFIG` | Override config file path |
| `JARVIS_LOG_LEVEL` | Override log level |
| `JARVIS_DASHBOARD_HOST` | Override dashboard bind host |
| `JARVIS_DASHBOARD_PORT` | Override dashboard bind port |
| `JARVIS_DASHBOARD_TOKEN` | Auth token for dashboard POST endpoints |
| `JARVIS_SHUTDOWN_TIMEOUT_S` | Graceful shutdown timeout |
| `GEMINI_API_KEY` | Enables Gemini routing and cloud fallback |
| `GROQ_API_KEY` | Optional cloud fallback provider |
| `OPENAI_API_KEY` | Optional cloud fallback provider |
| `ANTHROPIC_API_KEY` | Optional cloud fallback provider |
| `CLOUD_LLM_FALLBACK_ENABLED` | Enables or disables cloud fallback chain |
| `PORCUPINE_ACCESS_KEY` | Wake-word support for voice mode |

### Integration environment variables

| Integration | Variables |
| --- | --- |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| Google Calendar / Gmail | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` |
| Notion | `NOTION_API_KEY` |
| Spotify | `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REFRESH_TOKEN` |
| Email | `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`, `IMAP_HOST` |
| WhatsApp | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_FROM` |

## Integrations and tools

Integrations are auto-discovered from `integrations/clients/` and loaded only if their dependencies and credentials are available.

| Integration | Example tools | Notes |
| --- | --- | --- |
| Local calendar | `add_event`, `list_events` | Stores calendar data locally as ICS |
| Weather | `get_current_weather` | Public Open-Meteo based integration |
| Telegram | `send_telegram`, `get_updates` | Requires bot token and chat ID |
| Gmail | `list_unread`, `send_gmail`, `summarize_unread`, `mark_as_read` | Uses Google OAuth refresh token |
| Google Calendar | `create_event`, `list_events`, `delete_event`, `find_free_slot` | Uses Google OAuth refresh token |
| Notion | `create_page`, `query_database`, `append_block`, `get_page` | Requires Notion internal integration token |
| Spotify | `play_track`, `pause`, `search_track`, `get_current_track`, `create_playlist` | Uses Spotify refresh token |
| Email | `send_email`, `read_emails`, `search_emails` | IMAP/SMTP based |
| WhatsApp | `send_whatsapp` | Uses Twilio |

## Safety model

### Risk and confirmation

Jarvis does not treat all tools equally.

- Critical and forbidden actions can be blocked outright
- High-risk and user-confirmed actions require approval paths
- Read-oriented actions are allowed more freely
- Unknown tools default to conservative handling

The default config blocks or heavily restricts actions such as:

- Shell execution
- File deletion
- Registry writes
- Disk formatting and destructive disk operations
- GUI automation unless explicitly enabled

### Sandboxed execution

The execution layer is configured to operate within safe directories:

- `workspace`
- `outputs`
- `data`

This is controlled through the `execution.safe_directories` setting.

### Dashboard security

The dashboard serves read pages without auth, but POST endpoints such as `/command` require the `X-Dashboard-Token` header.

If you expose the dashboard beyond localhost:

- Set `JARVIS_DASHBOARD_TOKEN`
- Keep `allow_gui_automation = false` unless you explicitly need it
- Review all enabled integrations and outbound credentials

### Audit logging

Jarvis maintains a hash-chained append-only audit log in:

```text
logs/audit.jsonl
```

Use this to verify log integrity:

```powershell
python main.py --verify
```

Additional session artifacts are written under:

```text
outputs/Jarvis-Session/
```

## Data, logs, and runtime artifacts

| Path | Purpose |
| --- | --- |
| `data/jarvis_memory.db` | Primary SQLite memory store |
| `data/chroma/` | Chroma semantic memory store |
| `logs/app.log` | Application log |
| `logs/audit.jsonl` | Hash-chained audit log |
| `outputs/Jarvis-Session/` | Plans, traces, observations, reflections |
| `runtime/control_flags.json` | Dashboard/runtime control file |
| `memory/goals.json` | Goal persistence |
| `memory/calendar.ics` | Local calendar data |

Runtime paths are created automatically at startup when configured.

## Testing and quality

Run the full suite:

```powershell
pytest -q
```

Run the smoke suite for the entrypoint and startup flow:

```powershell
pytest tests/test_smoke.py -q
```

Current test coverage areas include:

- Runtime startup and shutdown behavior
- Risk evaluation and autonomy controls
- Hybrid memory and semantic recall
- Voice acceptance paths
- Tool dispatch and workflow execution
- Security boundaries and path traversal protection
- Dashboard-compatible controller lifecycle
- Integration behavior for Telegram, Gmail, Google Calendar, Notion, Spotify, and Weather

## Operational recommendations

For a production-style local deployment:

1. Keep Ollama running as a separate managed service.
2. Bind the dashboard to `127.0.0.1` unless remote access is intentional.
3. Set `JARVIS_DASHBOARD_TOKEN` before enabling dashboard POST actions.
4. Review `config/jarvis.ini` and shrink enabled capabilities before exposing integrations.
5. Keep `allow_gui_automation = false` until you have a clear approval path.
6. Run `python main.py --health-check` and `python main.py --verify` as part of operational validation.
7. Monitor `logs/app.log` and `logs/audit.jsonl`.

## Troubleshooting

### Health check reports Ollama unreachable

```powershell
ollama serve
ollama list
```

Also verify that the expected models are installed.

### Runtime fails because no model is available

Either:

- Pull the configured Ollama models, or
- Set `GEMINI_API_KEY` and optional cloud fallback keys

### An integration does not load

Check both:

- The required environment variables are present
- The integration's optional Python package is installed

The integration loader skips unavailable clients instead of crashing the runtime.

### Dashboard command requests return 401

Set a dashboard token and send it as:

```text
X-Dashboard-Token: <your-token>
```

### Voice mode fails to start

Verify audio dependencies and optional wake-word dependencies such as:

- `sounddevice`
- `soundfile`
- `whispercpp`
- `pvporcupine`

## Notes on repository maturity

- `main.py` and `core/controller_v2.py` are the current runtime path
- The repository still contains legacy modules and compatibility shims for earlier phases and tests
- The system is local-first, not cloud-only; however, some integrations and fallback providers require network access by design

## License

No project license file is currently present in the repository. Add one before distributing or deploying beyond personal/internal use.
