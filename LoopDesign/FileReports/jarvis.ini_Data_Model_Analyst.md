# Data Model Analyst Report for `jarvis.ini`

## File Information
- **Path:** `d:\AI\Jarvis\config\jarvis.ini`
- **Role:** Data Model Analyst

## Analysis
This file is the main initialization/configuration file containing various states, rules, APIs, and models across the system. It is structured using INI sections.

### Sections & Data Models

1. **`[general]`**
   - `name`: System name ("Jarvis")
   - `version`: System version ("2.0.0")
   - `environment`: Runtime env ("local")

2. **`[ollama]`**
   - `base_url`: Local LLM endpoint (`http://127.0.0.1:11434`)
   - `request_timeout_s`: Integer, timeout in seconds (`300`)

3. **`[models]`**
   - Defines tiers of models:
     - Tier 1 (Reflexive): `intent_model` (`qwen2.5:0.5b`), `summarize_model` (`llama3.2:1b`), `quick_model` (`gemma3:1b`)
     - Tier 2 (Execution): `chat_model` (`mistral:7b`), `tool_picker_model` (`mistral:7b`)
     - Tier 3 (Planning): `plan_model` (`deepseek-r1:8b`), `fallback_model` (`gemini-2.5-flash`), `vision_model` (`llava`)

4. **`[memory]`**
   - Filepaths for storage: `data_dir`, `sqlite_file`, `goals_file`, `chroma_dir`
   - Memory parameters: `embedding_model`, `llm_context_titles` (bool), `max_facts` (int), `semantic_top_k` (int), `stale_action_days` (int), `decay_cleanup_on_start` (bool)

5. **`[risk]`**
   - Maps action names to risk levels for the system state:
     - `forbidden_actions`, `blocked_actions`, `critical_actions`, `high_risk_actions`, `medium_risk_actions`, `low_risk_actions`
   - Defines execution and safety thresholds: `user_confirmed_actions`, `voice_confirm_threshold` (`MEDIUM`), `failsafe_auto_disable_on_error` (bool), `failsafe_error_threshold` (int)

6. **`[execution]`**
   - State variables governing execution boundaries: `safe_directories`, `max_read_bytes`, `allowed_apps`.
   - Flags for permissions: `allow_app_launch`, `allow_gui_automation`, `allow_web_search`, `sandboxed_execution`, `rollback_support`, `timeout_handling`, `stop_on_failure`, `rollback_on_failure`.
   - Integers: `step_timeout_s`, `max_step_workers`.

7. **`[web_search]`**
   - Features: `enabled`, `provider`, `summarize_results`, `auto_extract_query`.
   - Limits: `default_max_results`, `provider_timeout_s`, `scrape_timeout_s`, `quick_task_timeout_s`, `max_scrape_chars`.
   - Search parameters: `ddgs_region`, `ddgs_safesearch`, `tavily_api_key`.

8. **`[hardware]`**
   - Serial/Hardware state: `enabled`, `default_port`, `baud_rate`.

9. **`[logging]`**
   - Schema for logging paths: `log_dir`, `audit_file`, `app_file`, `trace_dir`.
   - Verbosity: `level` (`INFO`).

10. **`[voice]`**
   - STT configuration: `stt_engine`, `stt_model`, `stt_device`, `stt_compute_type`, `stt_silence_ms`, `stt_max_duration_s`, `stt_vad_aggressiveness`.
   - TTS configuration: `tts_engine`, `tts_voice`, `tts_streaming`, `tts_fallback_cli`.
   - Audio state: `listen_timeout_s`, `audio_sample_rate`, `audio_channels`, `audio_chunk_ms`, `wakeword_threshold`, `wakeword_model`, `wakeword_debounce_s`.
   - Vocabulary: `wake_word`, `cancel_words`.

11. **`[concurrency]`**
   - Limits: `max_parallel_tasks`.

12. **`[plugins]`**
   - Module resolution definitions: `directory`, `manifest_directory`, `enabled_scopes`.

13. **`[ai_os]`**
   - Application context states: `blueprint_file`, `workflow_catalog_dir`, `beginner_mode`, `advanced_mode`, `local_first`.

14. **`[proactive]`**
   - Background tasks criteria: `cpu_alert_threshold`, `ram_alert_threshold`, `goal_check_interval_minutes`.

15. **`[automation]`**
   - Drop box/folder watcher schema. Paths: `drop_root`, `commands_folder`, `rag_folder`, `processed_folder`, `failed_folder`, `screenshots_folder`, `recordings_folder`.
   - Chunking details for ingestion: `max_text_chars_per_item`, `chunk_size_chars`, `chunk_overlap_chars`.
   - State variables: `watch_screenshots`, `watch_recordings`, `live_screen_enabled`, `ingest_existing_on_start`.
   - Polling: `poll_interval_seconds`, `live_screen_interval_seconds`, `video_frame_interval_seconds`.
   - Runtime states: `ingest_log_file`, `state_file`.

16. **`[multi_agent]`**
   - Orchestration flags: `enabled`, `poll_interval_seconds`, `max_concurrent_workers`, `enable_interaction_agent`, `enable_web_agent`, `enable_desktop_agent`, `enable_rag_agent`, `enable_monitor_agent`.

17. **`[dashboard]`**
   - `control_file`: Path to UI-driven state overrides (`runtime/control_flags.json`).

18. **`[routing]`**
   - Inference execution schema: `strategy` (`adaptive`|`static`), `confidence_threshold`, `max_escalations`, `telemetry_enabled`, `telemetry_persistence`, `cost_preference`.

## API Contracts & Dependencies
- Depends on multiple local/remote LLMs endpoints (Ollama HTTP API port 11434).
- Dependent on Python modules/libraries for specific functionality config: Edge-TTS, Whisper, PyAudio, SQLite, Chroma, DDGS, Tavily, etc.

## Configuration Variables
Extensive variables documented in sections above.

## Prompts
No prompts explicitly found in this file.
