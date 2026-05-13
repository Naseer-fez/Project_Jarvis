# Jarvis — Local AI Assistant

A modular, privacy-first AI assistant that runs on your machine. Jarvis uses local LLMs via Ollama with automatic cloud fallback, and connects to your tools — Telegram, Spotify, Gmail, Google Calendar, Notion, GitHub, and more.

---

## Quick Start

### 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.11+ | 3.12 recommended |
| **Ollama** | Latest | [ollama.com/download](https://ollama.com/download) |
| **Git** | Any | For cloning the repo |



# Create virtual environment
python -m venv jarvis_env

# Activate it
# Windows PowerShell:
jarvis_env\Scripts\Activate.ps1
# Windows CMD:
jarvis_env\Scripts\activate.bat
# Linux/macOS:
source jarvis_env/bin/activate

# Install core dependencies
pip install -r requirements/base.txt
```

#### Optional Feature Groups

```bash
# Everything (all features)
pip install -r requirements/full.txt

# Or pick what you need:
pip install -r requirements/voice.txt          # Voice input/output
pip install -r requirements/desktop.txt        # Screen control & automation
pip install -r requirements/integrations.txt   # Telegram, GitHub SDKs
pip install -r requirements/dev.txt            # Testing & linting
```

### 3. Pull an Ollama Model

```bash
# Start Ollama (if not running)
ollama serve

# Pull the default chat model
ollama pull mistral:7b

# Optional: pull specialized models
ollama pull deepseek-r1:8b     # Planning & reasoning
ollama pull llama3.2:1b        # Fast summarization
ollama pull gemma3:1b          # Quick tasks
```

### 4. Configure Environment

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
```

Edit `.env` and add your API keys (see [API Keys Guide](#api-keys-guide) below).

### 5. Run

```bash
# Standard text mode
python main.py

# Text-only (explicitly no voice)
python Main_No_Voice.py

# With voice input/output
python main.py --voice

# With web dashboard at http://localhost:7070
python main.py --gui

# Health check (verify Ollama + config)
python main.py --health-check

# Windows PowerShell shortcut
.\run-jarvis.ps1
.\run-jarvis.ps1 --voice
.\run-jarvis.ps1 --gui
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                    (Entry Point)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              JarvisControllerV2                             │
│         (Orchestrates everything)                           │
├─────────────┬───────────────┬───────────────────────────────┤
│             │               │                               │
│  ┌──────────▼────────┐  ┌──▼──────────────┐  ┌────────────┐│
│  │   LLM Subsystem   │  │  Agent Loop     │  │   Tools    ││
│  │                    │  │                 │  │            ││
│  │  LLMClientV2      │  │  Plan → Risk    │  │ Web Search ││
│  │    ↓ ModelRouter   │  │  → Confirm      │  │ File I/O   ││
│  │    ↓ OllamaClient │  │  → Execute      │  │ Shell      ││
│  │    ↓ CloudLLMClient│  │  → Reflect      │  │ Desktop    ││
│  └───────────────────┘  └─────────────────┘  │ Screen     ││
│                                               └────────────┘│
├─────────────────────────────────────────────────────────────┤
│                     Integrations                            │
│  Telegram │ Spotify │ Gmail │ Calendar │ Notion │ GitHub    │
└─────────────────────────────────────────────────────────────┘
```

### LLM Subsystem (core/llm/)

Every prompt flows through one path:

1. **LLMClientV2** (`client.py`) — Public interface, all calls enter here
2. **ModelRouter** (`model_router.py`) — Picks the best model for the task type
3. **OllamaClient** (`ollama_client.py`) — Tries local Ollama first (free, private)
4. **CloudLLMClient** (`cloud_client.py`) — Falls back to cloud: Gemini → Groq → OpenAI → Anthropic

### Fallback Chains

| Component | Primary | Fallback 1 | Fallback 2 | Final |
|---|---|---|---|---|
| LLM | Ollama (local) | Gemini | Groq → OpenAI → Anthropic | Returns empty |
| TTS | edge-tts | pyttsx3 | Print to stdout | — |
| STT | Whisper (local) | Google Speech | Keyboard input | — |
| Web Search | Tavily | DuckDuckGo | — | Error message |

---

## API Keys Guide

All keys go in your `.env` file. **None are required** — Jarvis works fully offline with just Ollama. Add keys to unlock cloud features.

### LLM Providers (Cloud Fallback)

These power the cloud fallback when Ollama is unavailable.

#### Google Gemini (Recommended — Free Tier)
1. Go to [aistudio.google.com](https://aistudio.google.com/)
2. Click **"Get API Key"** → **"Create API Key"**
3. Copy the key

```env
GEMINI_API_KEY=AIzaSy...your_key_here
```

#### Groq (Fast & Free Tier)
1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up (free) → **"Create API Key"**
3. Copy the key

```env
GROQ_API_KEY=gsk_...your_key_here
```

#### OpenAI
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **"Create new secret key"**
3. Copy immediately (shown only once)

```env
OPENAI_API_KEY=sk-...your_key_here
```

#### Anthropic
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. **Settings** → **API Keys** → **"Create Key"**
3. Copy immediately

```env
ANTHROPIC_API_KEY=sk-ant-...your_key_here
```

---

### Integrations

#### Telegram Bot
1. Open Telegram, message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`, follow prompts → get your **bot token**
3. Message your bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your **chat ID**

```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=your_numeric_chat_id
```

#### Spotify
1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. **"Create App"** → set redirect URI to `http://localhost:8888/callback`
3. Copy Client ID and Client Secret
4. For the refresh token, run the OAuth flow once (see [Spotify Auth Guide](https://developer.spotify.com/documentation/web-api/tutorials/code-flow))

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
```

#### Google Calendar & Gmail
1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create a project → Enable **Calendar API** and **Gmail API**
3. **Credentials** → **Create OAuth 2.0 Client** (Desktop app)
4. Download the `credentials.json` file
5. Place it at `config/credentials.json`
6. On first run, Jarvis will open a browser for OAuth consent → generates `token.json`

```env
GOOGLE_CALENDAR_CREDENTIALS=config/credentials.json
GOOGLE_CALENDAR_TOKEN=config/calendar_token.json
GMAIL_CREDENTIALS=config/credentials.json
GMAIL_TOKEN=config/gmail_token.json
```

#### Notion
1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. **"New integration"** → give it a name → copy the **Internal Integration Token**
3. Share your Notion page/database with the integration

```env
NOTION_API_KEY=secret_...your_key_here
NOTION_DATABASE_ID=your_database_id
```

#### GitHub
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. **"Generate new token"** (classic) → select `repo` scope
3. Copy the token

```env
GITHUB_TOKEN=ghp_...your_token_here
```

#### Picovoice (Wake Word)
1. Go to [console.picovoice.ai](https://console.picovoice.ai/)
2. Sign up (free) → copy your **Access Key**

```env
PORCUPINE_ACCESS_KEY=your_access_key
```

---

## Configuration

The main config file is `config/jarvis.ini`. Key sections:

| Section | Controls |
|---|---|
| `[models]` | Which Ollama model handles each task type |
| `[memory]` | SQLite + ChromaDB paths, embedding model |
| `[risk]` | Blocked/high-risk/low-risk action lists |
| `[execution]` | Sandbox directories, app launch permissions |
| `[web_search]` | Search provider, result limits |
| `[voice]` | TTS/STT engines, wake word settings |
| `[hardware]` | Serial port for IoT devices |

### Model Tiers

Configured in `config/jarvis.ini` under `[models]`:

```ini
# Tier 1 — Reflexive (fast, <100ms)
intent_model = qwen2.5:0.5b
summarize_model = llama3.2:1b
quick_model = gemma3:1b

# Tier 2 — Execution (~500ms)
chat_model = mistral:7b

# Tier 3 — Reasoning (2-5s)
plan_model = deepseek-r1:8b
fallback_model = gemini-2.5-flash
```

---

## Project Structure

```
Jarvis/
├── main.py                    # Entry point
├── main_connector.py          # Bootstrap wiring
├── config/
│   └── jarvis.ini             # Runtime configuration
├── core/
│   ├── llm/                   # LLM subsystem (4 files)
│   │   ├── client.py          #   LLMClientV2 (public interface)
│   │   ├── ollama_client.py   #   OllamaClient (local HTTP)
│   │   ├── cloud_client.py    #   CloudLLMClient (Gemini→Groq→OpenAI→Anthropic)
│   │   └── model_router.py    #   ModelRouter (task→model mapping)
│   ├── agent/                 # Agent loop (plan→risk→act→reflect)
│   ├── tools/                 # Built-in tools (web, files, shell, screen)
│   ├── desktop/               # Desktop automation (mouse, keyboard)
│   ├── voice/                 # TTS, STT, wake word
│   ├── memory/                # SQLite + ChromaDB hybrid memory
│   ├── controller/            # Service wiring
│   └── autonomy/              # Risk evaluation, safety governor
├── integrations/
│   └── clients/               # Telegram, Spotify, Gmail, Calendar, etc.
├── dashboard/                 # Web UI (localhost:7070)
├── requirements/              # Split dependency groups
└── tests/                     # Test suite
```

---

## Running Tests

```bash
# Activate venv first, then:
pip install -r requirements/dev.txt

# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_cloud_llm_fallback.py -v
python -m pytest tests/test_runtime_model_routing.py -v
python -m pytest tests/test_core_lock.py -v
```

---

## Verify Installation

```bash
# Quick smoke test — checks LLM wiring
python -c "
import asyncio
from core.llm.client import LLMClientV2
c = LLMClientV2()
r = asyncio.run(c.complete('say hello', task_type='chat'))
print('PASS:', r[:60] if r else '(no LLM available — add Ollama or API keys)')
"

# Full health check
python main.py --health-check
```

---

## License

This project is for personal use. See repository for details.
