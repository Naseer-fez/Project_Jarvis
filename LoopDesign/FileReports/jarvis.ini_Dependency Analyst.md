# Dependency Analysis: jarvis.ini

## 1. Schema / API Contract
- Format: Standard INI file structure, parsed by configparser or similar utilities.
- Contains 18 logical sections: `general`, `ollama`, `models`, `memory`, `risk`, `execution`, `web_search`, `hardware`, `logging`, `voice`, `concurrency`, `plugins`, `ai_os`, `proactive`, `automation`, `multi_agent`, `dashboard`, `routing`.

## 2. Library Requirements / Service Dependencies
- **Ollama**: Exposes `http://127.0.0.1:11434`, crucial for running local LLMs.
- **SQLite**: Backend for structured memory logging (`data/jarvis_memory.db`).
- **ChromaDB**: Vector database required for semantic memory (`data/chroma`).
- **Tavily / DDGS (DuckDuckGo Search)**: Providers for the web search capabilities.
- **Whisper & Google**: STT engines.
- **edge-tts, pyttsx3, cli**: TTS engines.
- **Local Models**: 
  - Intent: `qwen2.5:0.5b`
  - Summarize: `llama3.2:1b`
  - Quick: `gemma3:1b`
  - Chat / Tool: `mistral:7b`
  - Plan: `deepseek-r1:8b`
  - Fallback: `gemini-2.5-flash`
  - Vision: `llava`
  - Embedding: `all-MiniLM-L6-v2`
  - Wakeword: `hey_jarvis`

## 3. Configuration Variables & Assumptions
- Defines extensive operational parameters, particularly around task limits and timeouts:
  - `stale_action_days`: 30
  - `max_facts`: 10000 (limits Chroma size)
  - `step_timeout_s`: 20
  - `max_read_bytes`: 200000
- **Risk Profiles**: Action whitelisting and blacklisting. Highly aggressive execution controls (e.g., `format_disk`, `wipe_disk` are forbidden, while `read_file` is medium risk).
- **Automation / Multi-Agent**: `enable_interaction_agent`, `enable_web_agent`, etc. all set to `true`.

## 4. Hidden Execution Links
- Heavily relies on file paths relative to the working directory:
  - `data/goals.json`
  - `workspace/jarvis_dropbox/*` (implies an automation pipeline that acts on dropped files)
  - `outputs/screenshots` and `outputs/screen_recordings` (implies background screenshot watchers)
  - `logs/audit.jsonl`
  - `runtime/control_flags.json`
  - `runtime/automation_state.json`
- `timeout_handling = true`, `rollback_on_failure = true`, `sandboxed_execution = true` imply wrappers or decorators govern the core execution path for safety.
- `routing.strategy = static` points to an escalation or failover mechanism via model tiers.
