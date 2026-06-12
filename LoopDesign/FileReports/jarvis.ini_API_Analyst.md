# API Analyst Report: jarvis.ini

## Target File
`d:\AI\Jarvis\config\jarvis.ini`

## Overview
`jarvis.ini` is the primary configuration file for the Jarvis agent. It defines settings for models, hardware, routing, logging, risk management, and overall system behavior. 

## Assumptions
- Parsed via a standard INI parser.
- Assumes local services (Ollama) are available at the specified `base_url`.
- Uses a tiered model approach (Reflexive, Execution, Planning).
- Memory features rely on local directories, SQLite, and Chroma.
- Relies on directories mapped out in `[execution]` and `[automation]` for workspace and dropping items.

## API Contracts & External Dependencies
- **Ollama API**: Assumes Ollama is running at `http://127.0.0.1:11434` with a default timeout of 300s. Models referenced (qwen2.5:0.5b, llama3.2:1b, gemma3:1b, mistral:7b, deepseek-r1:8b, llava) must be pulled and available.
- **Gemini API**: `fallback_model = gemini-2.5-flash` assumes availability of the Gemini API.
- **Tavily API**: Configurable web search fallback/provider.
- **DuckDuckGo Search**: Implied via `ddgs_region` and `ddgs_safesearch`.
- **Speech-To-Text / Text-To-Speech Engines**: Mentions `google`, `whisper`, `edge-tts`, `pyttsx3`, `cli`.
- **Wake Word**: `hey_jarvis` model with an implied connection to Porcupine or a similar wake word engine.

## Configuration Variables (Detailed)
### [general]
- `name`: Jarvis
- `version`: 2.0.0
- `environment`: local

### [ollama]
- `base_url`: http://127.0.0.1:11434
- `request_timeout_s`: 300

### [models]
- `intent_model`: qwen2.5:0.5b
- `summarize_model`: llama3.2:1b
- `quick_model`: gemma3:1b
- `chat_model`: mistral:7b
- `tool_picker_model`: mistral:7b
- `plan_model`: deepseek-r1:8b
- `fallback_model`: gemini-2.5-flash
- `vision_model`: llava

### [memory]
- Directory and DB Paths: `data_dir`, `sqlite_file`, `goals_file`, `chroma_dir`
- Retrieval/Storage: `embedding_model` (all-MiniLM-L6-v2), `llm_context_titles`, `max_facts`, `semantic_top_k`, `stale_action_days`, `decay_cleanup_on_start`

### [risk]
Defines explicit granular lists of functions across `forbidden_actions`, `blocked_actions`, `critical_actions`, `high_risk_actions`, `medium_risk_actions`, `low_risk_actions`, and `user_confirmed_actions`.
- Threshold settings: `voice_confirm_threshold`, `failsafe_auto_disable_on_error`, `failsafe_error_threshold`

### [execution]
- `safe_directories`, `max_read_bytes`, `allowed_apps`
- Feature flags: `allow_app_launch`, `allow_gui_automation`, `allow_web_search`, `sandboxed_execution`, `rollback_support`, `timeout_handling`, `stop_on_failure`, `rollback_on_failure`
- Execution limits: `step_timeout_s`, `max_step_workers`

### [web_search]
- Provider settings: `enabled`, `provider`, `default_max_results`, `summarize_results`, `auto_extract_query`
- Timeouts and constraints: `provider_timeout_s`, `scrape_timeout_s`, `quick_task_timeout_s`, `max_scrape_chars`
- DDG specific: `ddgs_region`, `ddgs_safesearch`
- Keys: `tavily_api_key`

### [hardware]
- `enabled`, `default_port`, `baud_rate` (implies Serial connection contract)

### [logging]
- `log_dir`, `audit_file`, `app_file`, `level`, `trace_dir`

### [voice]
- STT config: `enabled`, `wake_word`, `cancel_words`, `stt_engine`, `stt_model`, `stt_device`, `stt_compute_type`, `stt_silence_ms`, `stt_max_duration_s`, `stt_vad_aggressiveness`
- TTS config: `tts_engine`, `tts_voice`, `tts_streaming`, `tts_fallback_cli`
- Wake word config: `listen_timeout_s`, `audio_sample_rate`, `audio_channels`, `audio_chunk_ms`, `wakeword_threshold`, `wakeword_model`, `wakeword_debounce_s`

### [concurrency]
- `max_parallel_tasks`

### [plugins]
- `directory`, `manifest_directory`, `enabled_scopes`

### [ai_os]
- `blueprint_file`, `workflow_catalog_dir`, `beginner_mode`, `advanced_mode`, `local_first`

### [proactive]
- `cpu_alert_threshold`, `ram_alert_threshold`, `goal_check_interval_minutes`

### [automation]
- Directory watch configs for auto-execution and multimodal ingestion.
- `watch_screenshots`, `watch_recordings`, `live_screen_enabled`
- Text and video chunking sizes and rates: `max_text_chars_per_item`, `chunk_size_chars`, `chunk_overlap_chars`

### [multi_agent]
- Enablement of specialized agents: `interaction`, `web`, `desktop`, `rag`, `monitor`.

### [dashboard]
- `control_file`

### [routing]
- Strategy configurations: `strategy`, `confidence_threshold`, `max_escalations`, `telemetry_enabled`, `telemetry_persistence`, `cost_preference`
