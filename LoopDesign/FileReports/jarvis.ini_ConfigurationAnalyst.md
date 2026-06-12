# File Report: jarvis.ini
**Role:** Configuration Analyst

## File Overview
This is the core configuration file for the Jarvis system, containing granular settings for various modules, risk thresholds, models, and execution boundaries.

## Assumptions & Contracts
- **Format:** Standard INI configuration file.
- **Implicit Environment Assumptions:**
  - **Local Ollama:** Assumes an Ollama instance is running locally on `http://127.0.0.1:11434`.
  - **Paths:** Assumes the existence (or automatic creation) of various directories relative to the execution root, such as `data/`, `logs/`, `outputs/`, `workspace/`, `core/plugins/`, `plugins/`, `workflows/templates/`, and `runtime/`.
  - **Executables:** Assumes availability of `notepad`, `calc`, `mspaint`, `code` for allowed apps.
  - **STT/TTS Engines:** Assumes external libraries/APIs like `google`, `whisper`, `edge-tts`, `pyttsx3`, `cli` are available if voice features are enabled.

## Secrets & Env Vars
- Defines an empty key `tavily_api_key =` under `[web_search]`, which suggests it expects either an environment variable override or manual configuration.

## Extracted Prompts
- None.

## Configuration Variables
The file is structured into the following sections:
- `[general]`: App metadata (`name=Jarvis`, `version=2.0.0`, `environment=local`).
- `[ollama]`: Connection settings for Ollama (`base_url`, `request_timeout_s`).
- `[models]`: Segregates LLM models by execution tiers.
  - Tier 1 (Reflexive, <100ms): `intent_model=qwen2.5:0.5b`, `summarize_model=llama3.2:1b`, `quick_model=gemma3:1b`.
  - Tier 2 (Execution, ~500ms): `chat_model=mistral:7b`, `tool_picker_model=mistral:7b`.
  - Tier 3 (Planning, Heavy): `plan_model=deepseek-r1:8b`, `fallback_model=gemini-2.5-flash`, `vision_model=llava`.
- `[memory]`: Configures ChromaDB, SQLite paths, and embedding models (`all-MiniLM-L6-v2`). Also configures memory constraints (`max_facts=10000`, `semantic_top_k=5`, `stale_action_days=30`).
- `[risk]`: Classifies operations into lists such as `forbidden_actions`, `blocked_actions`, `critical_actions`, `high_risk_actions`, `medium_risk_actions`, and `low_risk_actions`. It defines `user_confirmed_actions`, voice thresholds (`voice_confirm_threshold=MEDIUM`), and fail-safe triggers.
- `[execution]`: Controls system interactions, sandboxing, auto-rollback behavior, worker limits, and safe directories (`workspace`, `outputs`, `data`).
- `[web_search]`: Controls web search params (`ddgs_region`, timeout limits, auto extraction).
- `[hardware]`: Serial port configs. Currently `enabled = false`.
- `[logging]`: Logging formats, levels (`INFO`), and output directories.
- `[voice]`: STT/TTS engine selection, timeouts, wake words (`jarvis`), and cancel words.
- `[concurrency]`: Defines `max_parallel_tasks=3`.
- `[plugins]`: Points to plugin manifest paths and enabled scopes.
- `[ai_os]`: Links to the blueprint JSON file and workflow templates.
- `[proactive]`: Thresholds for CPU/RAM alerts (both `90`).
- `[automation]`: Defines the drop folder structure (`jarvis_dropbox`) for automation triggers, rag document ingestion chunking limits, and screen recording intervals.
- `[multi_agent]`: Enablement flags for background subagents (web, desktop, rag, monitor).
- `[dashboard]`: Defines the `control_file` (`runtime/control_flags.json`).
- `[routing]`: Defines LLM fallback routing parameters (`strategy=static`, `confidence_threshold=0.7`, `cost_preference=balanced`).
