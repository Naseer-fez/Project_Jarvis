# File Report: jarvis.ini
**Role**: Documentation Analyst

## 1. Assumptions
- This is the main configuration file for the Jarvis agent. It uses a standard INI format.
- The system heavily relies on local model execution via Ollama (default URL provided).
- There is a 3-tier model strategy: Tier 1 (Reflexive), Tier 2 (Execution), Tier 3 (Planning).
- A robust risk-management strategy categorizes actions by severity (low, medium, high, critical, forbidden).
- The system has multi-agent capabilities, a background automation watcher (recordings/screenshots), voice capabilities, hardware integrations, and plugin support.

## 2. Schema
The INI schema is divided into the following sections:
- `[general]`: Basic system details.
- `[ollama]`: Local model runner settings.
- `[models]`: Model names assigned to functional roles.
- `[memory]`: Paths and settings for vector (Chroma) and relational (SQLite) DBs.
- `[risk]`: Granular permission sets for tools/actions.
- `[execution]`: Workspace bounding and runtime safeguards.
- `[web_search]`: Settings for search providers and scraping.
- `[hardware]`: Serial port/hardware connection defaults.
- `[logging]`: Paths and log levels.
- `[voice]`: STT, TTS, and wakeword configuration.
- `[concurrency]`: Task parallelism limits.
- `[plugins]`: Plugin directories and scopes.
- `[ai_os]`: AI OS specifics like blueprint files.
- `[proactive]`: Alert thresholds for system monitoring.
- `[automation]`: Dropbox/watcher folder paths and processing params.
- `[multi_agent]`: Sub-agent toggles and pooling.
- `[dashboard]`: UI control files.
- `[routing]`: Model routing strategies and fallback thresholds.

## 3. API Contracts
- Connects to an Ollama server locally at `http://127.0.0.1:11434`.
- Potentially connects to Tavily API if the key is provided in the web_search section (or environment vars).

## 4. Dependencies
- **Ollama**: For local model execution.
- **SQLite**: For memory and task persistence.
- **ChromaDB**: For vector search and RAG.
- **Whisper/Google STT**, **Edge-TTS/pyttsx3**: Voice integrations.
- **Local Filesystem**: Requires specific folder structures (`data/`, `logs/`, `workspace/`, `outputs/`).

## 5. Configuration Variables
- Models: `qwen2.5:0.5b`, `llama3.2:1b`, `gemma3:1b`, `mistral:7b`, `deepseek-r1:8b`, `gemini-2.5-flash`, `llava`.
- Risk: Defines explicit lists of `forbidden_actions`, `blocked_actions`, `critical_actions`, `high_risk_actions`, `medium_risk_actions`, `low_risk_actions`, `user_confirmed_actions`.
- Execution bounds: `max_read_bytes = 200000`, `step_timeout_s = 20`, `safe_directories`.
- Automation bounds: `chunk_size_chars = 1200`, `chunk_overlap_chars = 120`.
- Routing: `strategy = static`, `confidence_threshold = 0.7`.

## 6. Prompts
- None found explicitly as raw text prompts in this config file.
