# 08 Configuration Subsystem: Deep Semantic Analysis

## 1. Intent: WHY does this subsystem exist?
The Configuration Subsystem exists to decouple the physical capabilities and boundaries of the autonomous AI from its core execution logic. In a tiered multi-agent environment where an AI acts upon a local operating system, hardcoding environmental constraints, model mappings, and risk guardrails is catastrophically dangerous. This subsystem provides a strict, deterministic contract for the agent's identity, constraints, capabilities, and dependencies, ensuring that operational parameters can be adjusted safely without altering the system's code.

## 2. Responsibility: WHAT responsibility does it own?
The Configuration Subsystem acts as the supreme governor of the system's operational parameters. Its responsibilities are split across state, security, and behavior:
*   **Execution Stratification:** It maps conceptual task complexity to specific compute resources by defining Tier 1 (Intent/Reflex), Tier 2 (Execution), and Tier 3 (Planning) LLM models. 
*   **Boundary & Risk Enforcement:** It holds the definitive matrix of allowed operations, classifying capabilities into strict risk strata (from `forbidden_actions` to `low_risk_actions`) and dictating exactly which operations mandate human-in-the-loop (HITL) confirmation.
*   **System Identification & Persona:** It manages the semantic state of the interaction environment (e.g., user profiles, interaction styles, timezones) which dictate how the LLMs are prompted globally.
*   **Resource Allocation:** It governs concurrency thresholds (e.g., maximum parallel background tasks), memory constraints, and sandbox limitations.
*   **Integration Binding:** It acts as the exclusive custodian for all secrets, credentials, and API endpoints connecting the agent to the outside world.

## 3. Interaction: HOW does it interact with the rest of the system?
*   **Agent Loop Initializer:** At system startup, configuration matrices parameterize the central routing engines, determining whether requests follow static routing or adaptive LLM fallback protocols.
*   **Safety Interceptor:** Before any tool or command is invoked, the execution engine queries the configuration's risk matrix to evaluate whether the proposed action violates sandboxed workspace paths or crosses a threshold requiring voice or explicit user approval.
*   **Subagent Workload Manager:** Background multi-agent processes poll the configuration to enforce rate limits, web search constraints, and filesystem extraction pathways.
*   **Context Injector:** Memory and routing layers dynamically ingest parameters like `communication_style`, `expertise_level`, and `timezone` to continuously augment the prompt parameters injected into all LLM tiers.

## 4. Criticality: WHAT would break if removed?
Removing the Configuration Subsystem would result in a total collapse of both the system's safety and its computational efficiency:
*   **Catastrophic Safety Failure:** The agent would lose its sandboxed boundaries and risk classifications. It would either execute destructive system commands blindly or become entirely paralyzed by default-deny fallbacks.
*   **Model Routing Collapse:** The tiered intelligence system would fail, routing simple reflexive tasks to expensive reasoning models or sending complex execution logic to lightweight models incapable of resolving them.
*   **Loss of Persona & Context:** All stateful tracking of the user's workflow, communication preferences, and goals would vanish, reverting the system to an amnesiac base state.
*   **Integration Disconnection:** All hooks into email, cloud fallbacks, Smart Home systems, and messaging layers would immediately fault due to unresolvable credential variables.
*   **Resource Exhaustion:** Without concurrency and timeout rules, background task automation would spawn unbounded processes, rapidly consuming all hardware resources.

## 5. Reconstruction Strategy: HOW would it be rebuilt from scratch?
If rebuilding this subsystem without source code, it must be constructed as a rigid schema-validation engine rather than a loose collection of variables:

*   **Phase 1: Environment & Secrets Broker:** Build an OS-level environment variable loader that exclusively handles secrets (tokens, API keys). Expose this to the rest of the system as an immutable, read-only dependency injected at startup.
*   **Phase 2: Risk Matrix Engine:** Construct a rigid, declarative matrix mapping all possible system actions to risk categories (Forbidden, Critical, High, Medium, Low). Attach an approval-gate interceptor that halts execution unless the necessary HITL constraints are met.
*   **Phase 3: Tiered Model Router:** Define schemas for `[models]` and `[routing]` that decouple the concept of an "operation tier" from a specific model string. Implement logic that evaluates a task's requirement (cost, speed, capability) and dynamically routes it to the designated Tier 1, 2, or 3 endpoints.
*   **Phase 4: Concurrency & Boundary Enforcement:** Implement application-level limits governing maximum worker threads, explicit allowed directories (`workspace/`, `outputs/`), and whitelisted external applications.
*   **Phase 5: Dynamic State Stores:** Implement a hot-reloadable JSON storage mechanism (`user_profile.json`, `goals.json`) to persist context across sessions, tracking explicit configuration (e.g., timezone) and implicit state (interaction counts). Bind system prompts directly to this state.

## 6. Literal Configuration Schemas

### `settings.env` Keys
```env
# Email
EMAIL_ADDRESS=
EMAIL_PASSWORD=
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
IMAP_HOST=imap.gmail.com

# WhatsApp via Twilio
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# Home Assistant
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=

# GitHub
GITHUB_TOKEN=
GITHUB_DEFAULT_REPO=

PORCUPINE_ACCESS_KEY=
CALENDAR_ICS_PATH=data/calendar.ics

# Cloud LLM Fallback
GROQ_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
CLOUD_LLM_FALLBACK_ENABLED=true
```

### `jarvis.ini` Structure
```ini
[general]
name = Jarvis
version = 2.0.0
environment = local

[ollama]
base_url = http://127.0.0.1:11434
request_timeout_s = 300

[models]
# Tier 1: Reflexive tasks (Lightning fast, < 100ms)
intent_model = qwen2.5:0.5b
summarize_model = llama3.2:1b
quick_model = gemma3:1b

# Tier 2: Execution & Tools (Fast, ~500ms)
chat_model = mistral:7b
tool_picker_model = mistral:7b

# Tier 3: Planning & Reasoning (Heavy, ~2s - 5s+)
plan_model = deepseek-r1:8b
fallback_model = gemini-2.5-flash
vision_model = llava

[memory]
data_dir = data
sqlite_file = data/jarvis_memory.db
goals_file = data/goals.json
chroma_dir = data/chroma
embedding_model = all-MiniLM-L6-v2
llm_context_titles = true
max_facts = 10000
semantic_top_k = 5
stale_action_days = 30
decay_cleanup_on_start = true

[risk]
# Risk levels: LOW, MEDIUM, HIGH, CRITICAL
forbidden_actions = format_disk, wipe_disk, registry_write
blocked_actions = shell_exec, file_delete
critical_actions = execute_shell, delete_file
high_risk_actions = write_file, write_file_safe, file_write, process_spawn, launch_application, app_open, click, double_click, right_click, click_text_on_screen, click_screen_target, double_click_screen_target, right_click_screen_target, type_text, hotkey, press_key, move_mouse, drag, focus_window, scroll, send_hardware_command, serial_send
medium_risk_actions = read_file, file_read, web_search, web_scrape, capture_screen, capture_region, find_text_on_screen, read_screen_text, wait_for_text_on_screen, describe_screen, get_active_window, read_sensor, sensor_read, notification
low_risk_actions = search_memory, log_event, memory_read, memory_write, speak, display, status, health_check, get_time, get_system_stats, list_directory, system_stats, vision_analyze
user_confirmed_actions = launch_application, app_open, click, double_click, right_click, click_text_on_screen, click_screen_target, double_click_screen_target, right_click_screen_target, type_text, hotkey, press_key, move_mouse, drag, focus_window, scroll, send_hardware_command, serial_send
voice_confirm_threshold = MEDIUM
failsafe_auto_disable_on_error = true
failsafe_error_threshold = 3

[execution]
safe_directories = workspace,outputs,data
max_read_bytes = 200000
allowed_apps = notepad,calc,mspaint,code
allow_app_launch = true
allow_gui_automation = true
allow_web_search = true
sandboxed_execution = true
rollback_support = true
timeout_handling = true
step_timeout_s = 20
stop_on_failure = false
rollback_on_failure = true
max_step_workers = 4

[web_search]
enabled = true
provider = auto
default_max_results = 5
summarize_results = true
auto_extract_query = true
provider_timeout_s = 8
scrape_timeout_s = 10
quick_task_timeout_s = 4
max_scrape_chars = 8000
ddgs_region = wt-wt
ddgs_safesearch = moderate
tavily_api_key =

[hardware]
enabled = false
default_port = 
baud_rate = 9600

[logging]
log_dir = logs
audit_file = logs/audit.jsonl
app_file = logs/app.log
level = INFO
trace_dir = outputs/Jarvis-Session

[voice]
enabled = false
wake_word = jarvis
cancel_words = cancel,stop,never mind,abort
stt_engine = google, whisper
stt_model = base
stt_device = cpu
stt_compute_type = int8
stt_silence_ms = 500
stt_max_duration_s = 30
stt_vad_aggressiveness = 2
tts_engine = edge-tts, pyttsx3, cli
tts_voice = en-US-GuyNeural
tts_streaming = true
tts_fallback_cli = true
listen_timeout_s = 8
audio_sample_rate = 16000
audio_channels = 1
audio_chunk_ms = 30
wakeword_threshold = 0.5
wakeword_model = hey_jarvis
wakeword_debounce_s = 1.0

[concurrency]
max_parallel_tasks = 3

[plugins]
directory = core/plugins
manifest_directory = plugins
enabled_scopes = core

[ai_os]
blueprint_file = config/ai_os.json
workflow_catalog_dir = workflows/templates
beginner_mode = true
advanced_mode = true
local_first = true

[proactive]
cpu_alert_threshold = 90
ram_alert_threshold = 90
goal_check_interval_minutes = 5

[automation]
enabled = true
auto_execute_commands = true
drop_root = workspace/jarvis_dropbox
commands_folder = workspace/jarvis_dropbox/commands
rag_folder = workspace/jarvis_dropbox/rag
processed_folder = workspace/jarvis_dropbox/processed
failed_folder = workspace/jarvis_dropbox/failed
watch_screenshots = true
screenshots_folder = outputs/screenshots
watch_recordings = true
recordings_folder = outputs/screen_recordings
live_screen_enabled = true
poll_interval_seconds = 3
live_screen_interval_seconds = 20
video_frame_interval_seconds = 2
max_video_samples = 20
max_text_chars_per_item = 12000
chunk_size_chars = 1200
chunk_overlap_chars = 120
min_file_age_seconds = 2
ingest_existing_on_start = false
ingest_log_file = runtime/automation_ingest.jsonl
state_file = runtime/automation_state.json

[multi_agent]
enabled = true
poll_interval_seconds = 5
max_concurrent_workers = 5
enable_interaction_agent = true
enable_web_agent = true
enable_desktop_agent = true
enable_rag_agent = true
enable_monitor_agent = true

[dashboard]
control_file = runtime/control_flags.json

[routing]
# Strategy: "adaptive" (cost-optimizing) or "static" (legacy tier lookup)
strategy = static

# Minimum reliability threshold before escalating (0.0-1.0)
confidence_threshold = 0.7

# Maximum escalation attempts per request
max_escalations = 1

# Enable execution telemetry recording
telemetry_enabled = true

# Telemetry persistence: "memory" (session only) or "file" (JSONL to logs/)
telemetry_persistence = memory

# Cost preference: "minimum" (always cheapest), "balanced" (cost vs quality), "quality" (best available)
cost_preference = balanced
```
