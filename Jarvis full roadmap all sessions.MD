JARVIS — COMPLETE CAPABILITY ROADMAP
All Future Sessions — Full Instructions for Each
MASTER CAPABILITY MAP
PHASE 6  (Next)   → GUI Automation + Screen Understanding
PHASE 7           → Proactive Intelligence + Scheduling + Goals
PHASE 8           → User Profiling + Personalization + Synthesis
PHASE 9           → Dashboard + Web UI + Remote Control
PHASE 10          → Hardware / IoT + Serial Control
PHASE 11          → Multi-Model + Model Switching + Local Fine-tuning
PHASE 12          → Security Hardening + Full Test Suite
SESSION 6 — GUI AUTOMATION + SCREEN UNDERSTANDING
Goal: Jarvis can see your screen, click, type, and control desktop apps.

Prompt to paste at session start:

You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-5 complete.

INSTALL BEFORE STARTING:
  pip install pyautogui pillow pygetwindow pytesseract
  # Also install Tesseract OCR for Windows from https://github.com/UB-Mannheim/tesseract/wiki

THIS SESSION GOAL: GUI Automation — Jarvis can see the screen and control the desktop.

FILES TO CREATE:
  core/tools/screen.py
    - capture_screen() — PIL screenshot, return base64 PNG + dimensions
    - capture_region(x, y, width, height) — screenshot of specific area
    - find_text_on_screen(text) — OCR via pytesseract, return list of (text, x, y, w, h)
    - find_image_on_screen(template_path) — pyautogui.locate, return center coords or None
    - describe_screen(llm_client) — capture + send to LLaVA, return natural language description

  core/tools/gui_control.py
    - click(x, y, button="left") — move + click with pre-click screenshot for audit
    - double_click(x, y)
    - right_click(x, y)
    - type_text(text, interval=0.05) — type with human-like interval
    - hotkey(*keys) — pyautogui.hotkey wrapper
    - scroll(x, y, clicks=3, direction="down")
    - drag(x1, y1, x2, y2, duration=0.5)
    - get_active_window() — return {"title": str, "x": int, "y": int, "width": int, "height": int}
    - focus_window(title_substring) — bring window to front
    - All functions save before/after screenshots to outputs/gui_audit/

  core/tools/app_launcher.py
    - launch_app(name_or_path) — launch by name (notepad, chrome) or full path
    - close_app(title_substring) — close window by title
    - list_open_windows() — return list of window titles
    - wait_for_window(title_substring, timeout=10) — wait until window appears

FILE MODIFICATIONS:
  core/autonomy/risk_evaluator.py:
    - Add to _DEFAULT_LOW: "capture_screen", "capture_region", "describe_screen",
      "find_text_on_screen", "get_active_window", "list_open_windows"
    - Add to _DEFAULT_CONFIRM: "click", "double_click", "type_text", "hotkey",
      "scroll", "drag", "focus_window", "launch_app", "close_app"
  core/tools/builtin_tools.py:
    - Register all new screen and gui tools in register_all_tools()
  core/llm/task_planner.py:
    - Add screen and GUI tool schemas to SYSTEM_TOOL_SCHEMA

SAFETY RULES (non-negotiable):
  - Every click and type saves a before screenshot to outputs/gui_audit/
  - NEVER allow clicks on coordinates outside the current screen bounds
  - Add a 0.3 second delay before every click — gives user time to intervene
  - Hotkey Ctrl+Shift+J = emergency interrupt — abort all GUI actions immediately
  - If pyautogui.FAILSAFE triggers (mouse to corner): treat as user_interrupt
  - Never allow type_text to type passwords — if content contains "password": refuse

DO NOT:
  - Do not use win32api or ctypes directly — use pyautogui only
  - Do not automate browser login forms
  - Do not run GUI automation if voice mode is active (screen reader conflict)
  - Do not click faster than 1 click per 300ms — protect against runaway loops

PATHS:
  core/tools/screen.py        — new file
  core/tools/gui_control.py   — new file
  core/tools/app_launcher.py  — new file

VERIFICATION:
  python -c "from core.tools.screen import capture_screen; print('OK')"
  python -c "from core.tools.gui_control import click; print('OK')"
SESSION 7 — PROACTIVE INTELLIGENCE + SCHEDULER + GOALS
Goal: Jarvis runs in the background, pursues goals, sends reminders, and acts without being asked.

Prompt to paste at session start:

You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-6 complete.

INSTALL BEFORE STARTING:
  pip install apscheduler

THIS SESSION GOAL: Proactive Intelligence — Jarvis can schedule tasks, pursue goals, and notify.

FILES TO CREATE:
  core/autonomy/goal_manager.py (REPLACE existing stub)
    - GoalManager class
    - add_goal(description, priority=1, due_datetime=None, repeat_interval=None) → goal_id
    - complete_goal(goal_id)
    - fail_goal(goal_id, reason)
    - get_active_goals() → list
    - get_overdue_goals() → list
    - Persist goals to memory/goals.json — load on init
    - Priority levels: 1=LOW, 2=NORMAL, 3=HIGH, 4=URGENT

  core/agentic/scheduler.py (REPLACE existing stub)
    - JarvisScheduler class wrapping APScheduler AsyncIOScheduler
    - schedule_once(func, run_at: datetime, job_id=None)
    - schedule_recurring(func, interval_seconds, job_id=None)
    - schedule_daily(func, hour, minute, job_id=None)
    - cancel(job_id)
    - list_jobs() → list of {job_id, next_run, func_name}
    - start() and shutdown() — must be called with await
    - On job failure: log to audit, do NOT crash the scheduler

  core/proactive/notifier.py   — NEW FILE
    - NotificationManager class
    - notify(message, level="info") — print to terminal + TTS if voice enabled
    - notify_windows(title, message) — Windows 10 toast notification via win10toast or plyer
    - schedule_reminder(message, in_seconds) — schedule a one-shot notification
    - If win10toast not installed: fall back to terminal print

  core/proactive/background_monitor.py   — NEW FILE
    - BackgroundMonitor class
    - monitor_file(path, on_change_callback) — watch for file changes
    - monitor_system_resources(cpu_threshold=90, ram_threshold=90) — alert if exceeded
    - Uses asyncio.sleep polling (1s interval) — no watchdog dependency
    - All monitors run as asyncio tasks, cancellable via stop()

FILE MODIFICATIONS:
  core/agent/controller.py (MainController):
    - Add self.goal_manager = GoalManager() in __init__
    - Add self.scheduler = JarvisScheduler() in __init__
    - Add self.notifier = NotificationManager() in __init__
    - In run(): call await self.scheduler.start() before loop, await self.scheduler.shutdown() after
    - Add intent routing: "remind me", "schedule", "set goal" → goal_manager + scheduler
    - Every N minutes (configurable in jarvis.ini): check overdue goals, notify user

  config/jarvis.ini — add section:
    [proactive]
    goal_check_interval_minutes = 5
    enable_background_monitor = true
    cpu_alert_threshold = 90
    ram_alert_threshold = 90

DO:
  - Goals must survive app restart — persist to memory/goals.json
  - Scheduler must recover from APScheduler job failures without crashing
  - NotificationManager must degrade gracefully if win10toast not installed
  - Background monitor tasks must be cancelled cleanly on shutdown

DO NOT:
  - Do not auto-execute goals — only notify the user that a goal is due
  - Do not run background tasks if autonomy_level < 2
  - Do not schedule jobs with intervals under 30 seconds (spam prevention)
  - Do not import apscheduler at module level — wrap in try/except ImportError

PATHS:
  core/autonomy/goal_manager.py          — replace stub
  core/agentic/scheduler.py              — replace stub
  core/proactive/notifier.py             — new file
  core/proactive/background_monitor.py   — new file
  core/proactive/__init__.py             — new (empty init)

VERIFICATION:
  python -c "from core.autonomy.goal_manager import GoalManager; print('OK')"
  python -c "from core.agentic.scheduler import JarvisScheduler; print('OK')"
  python -c "from core.proactive.notifier import NotificationManager; print('OK')"
  python main.py --help
SESSION 8 — USER PROFILING + PERSONALIZATION + SYNTHESIS
Goal: Jarvis learns who you are, adapts its tone, and remembers your preferences across sessions.

Prompt to paste at session start:

You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-7 complete.

THIS SESSION GOAL: User Profiling — Jarvis learns and adapts to the specific user over time.

FILES TO MODIFY/REPLACE:
  core/profile.py (REPLACE — keep class name UserProfileEngine):
    - Load from memory/user_profile.json on init
    - Save to disk on every update (atomic write via temp file + rename)
    - Fields: name, communication_style (formal/casual/technical), expertise_level,
      preferred_topics, disliked_topics, timezone, language, preferred_model,
      interaction_count, first_seen, last_seen
    - update_from_conversation(user_text, jarvis_response) — extract signals
    - get_system_prompt_injection() → str — compact profile for LLM context
    - get_communication_style() → str — "Be concise." or "Be detailed." etc.

  core/synthesis.py (REPLACE — keep class name ProfileSynthesizer):
    - synthesize(memory_entries: list, profile: UserProfileEngine) → delta dict
    - Calls LLM with SYNTHESIS_SYSTEM prompt
    - Apply delta to profile — only update fields with confidence > 0.6
    - Run synthesis after every 20 new conversations (configurable)
    - Save synthesis run timestamp to profile

  core/metrics/confidence.py (WIRE INTO CONTROLLER):
    - Already has ConfidenceModel — use it in agent_loop
    - Update signals: intent_clarity from IntentClassifierV2 confidence score
    - Update tool_reliability from ToolObservation success rate
    - If score < 0.4: ask user for clarification before executing
    - If score > 0.8: execute without asking (if autonomy allows)

FILE MODIFICATIONS:
  core/llm/client.py:
    - In _build_system(): call profile.get_system_prompt_injection() if profile available
    - Add self.profile = None to __init__, set via set_profile(profile)
  core/agent/controller.py (MainController):
    - Add self.profile = UserProfileEngine() in __init__
    - Add self.synthesizer = ProfileSynthesizer(self.llm) in __init__
    - After every conversation: call profile.update_from_conversation()
    - Every 20 conversations: call await synthesizer.synthesize(memory_entries, profile)
    - Pass profile to llm client: self.llm.set_profile(self.profile)
  core/controller_v2.py (JarvisControllerV2):
    - Add same profile engine wiring

DO:
  - Profile file must be human-readable JSON — the user may hand-edit it
  - Atomic writes only — never corrupt the profile on crash (write to .tmp then rename)
  - The LLM prompt injection from profile must be under 200 tokens — keep it compact
  - Synthesis must run in a background asyncio task — never block the main loop

DO NOT:
  - Do not store sensitive personal data (health, financial) in the profile
  - Do not auto-update profile from single messages — require pattern across 3+ interactions
  - Do not let the profile override user's explicit instructions in a conversation
  - Do not call the synthesizer more than once every 20 conversations — it's expensive

PATHS:
  core/profile.py      — replace
  core/synthesis.py    — replace
  core/metrics/confidence.py — wire into controller

VERIFICATION:
  python -c "from core.profile import UserProfileEngine; p = UserProfileEngine(); print(p.get_communication_style())"
  python -c "from core.synthesis import ProfileSynthesizer; print('OK')"
SESSION 9 — DASHBOARD + WEB UI + REMOTE CONTROL
Goal: A local web dashboard to monitor Jarvis, view logs, manage memory, and send commands remotely.

Prompt to paste at session start:

You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-8 complete.

INSTALL BEFORE STARTING:
  pip install fastapi uvicorn websockets jinja2 python-multipart

THIS SESSION GOAL: Web Dashboard — local web UI to monitor and control Jarvis.

DO NOT use React, Node, npm, or any frontend build tools.
Use FastAPI + Jinja2 templates + vanilla JavaScript only.
All in a single self-contained dashboard module.

FILES TO CREATE:
  dashboard/
    __init__.py
    server.py          — FastAPI app, all routes, WebSocket handler
    static/
      style.css        — minimal dark theme CSS
      app.js           — vanilla JS for WebSocket updates
    templates/
      index.html       — main dashboard page
      memory.html      — memory browser
      goals.html       — active goals view
      logs.html        — audit log viewer

  dashboard/server.py ROUTES:
    GET  /             — main dashboard (state, last response, session info)
    GET  /memory       — browse and search memory
    GET  /goals        — view/add/complete goals
    GET  /logs         — paginated audit log viewer
    GET  /health       — JSON health check (for monitoring)
    POST /command      — send a command to Jarvis (same as typing in CLI)
    POST /memory/add   — manually add a memory entry
    POST /goals/add    — manually add a goal
    WS   /ws           — WebSocket: server pushes state updates every 2s

  dashboard/server.py WEBSOCKET:
    Push every 2 seconds:
    {"state": "IDLE", "last_response": "...", "session_id": "...",
     "memory_count": N, "active_goals": N, "ollama_online": bool}

  main.py MODIFICATION:
    Add --dashboard flag: python main.py --dashboard
    Starts uvicorn dashboard.server:app on localhost:7070 in background thread
    Dashboard and CLI run simultaneously

DO:
  - Dashboard ONLY binds to 127.0.0.1 — never 0.0.0.0 (security)
  - All POST endpoints require a simple token from config/jarvis.ini [dashboard] secret
  - WebSocket drops connection gracefully if Jarvis is shutting down
  - Memory browser must support search by keyword
  - Dark theme — the user is likely using this at night

DO NOT:
  - Do not expose any file system browsing via the API
  - Do not allow dashboard to trigger DELETE or high-risk actions — read + command only
  - Do not use SQLAlchemy or any ORM — read directly from memory's SQLite file
  - Do not install tailwind, webpack, or any build tool
  - Do not run dashboard on any port other than 7070 (configurable in jarvis.ini)

PATHS:
  dashboard/__init__.py      — new
  dashboard/server.py        — new
  dashboard/static/          — new dir
  dashboard/templates/       — new dir
  main.py                    — add --dashboard flag

VERIFICATION:
  python -c "from dashboard.server import app; print('OK')"
  python main.py --dashboard &
  curl http://localhost:7070/health
SESSION 10 — HARDWARE / IoT + SERIAL CONTROL
Goal: Jarvis controls physical hardware — Arduino, sensors, lights, motors via serial.

Prompt to paste at session start:

You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-9 complete.

INSTALL BEFORE STARTING:
  pip install pyserial

THIS SESSION GOAL: Hardware Integration — connect Jarvis to physical devices via serial port.

THE EXISTING FILE:
  core/hardware/serial_controller.py — already has SerialController class with simulation mode
  Read it fully before writing anything.

FILES TO MODIFY:
  core/hardware/serial_controller.py — EXTEND (do NOT rewrite what works):
    - Add async wrapper: async_send_command(cmd, value) — wraps sync send in run_in_executor
    - Add sensor_read_loop(callback, interval=1.0) — async loop, calls callback(sensor_data)
    - Add firmware_ping() → bool — send PING, expect PONG within 2s
    - Simulation mode must produce realistic fake sensor values (temperature 20-25, humidity 40-60)

FILES TO CREATE:
  core/hardware/device_registry.py — NEW:
    - DeviceRegistry class
    - register_device(name, com_port, baud_rate, device_type)
    - get_device(name) → SerialController instance (cached)
    - list_devices() → list of {name, port, connected, simulation_mode}
    - Persist device list to config/devices.json

  core/tools/hardware_tools.py — NEW:
    - send_hardware_command(device_name, command, value) → ToolResult
    - read_sensor(device_name, sensor_type) → ToolResult
    - list_hardware_devices() → ToolResult
    - ping_device(device_name) → ToolResult
    - All use DeviceRegistry to find device — never hardcode port names

FILE MODIFICATIONS:
  core/autonomy/risk_evaluator.py:
    - Add to _DEFAULT_LOW: "list_hardware_devices", "ping_device", "read_sensor"
    - Add to _DEFAULT_CONFIRM: "send_hardware_command"
  core/tools/builtin_tools.py:
    - Register hardware tools in register_all_tools()
  core/llm/task_planner.py:
    - Add hardware tool schemas

DO:
  - Simulation mode is the default — never require real hardware to start
  - All serial reads have a 2s timeout — never block forever
  - Commands sent to hardware are logged to audit BEFORE being sent
  - If serial port disconnects mid-session: log warning, enter simulation mode, do NOT crash

DO NOT:
  - Do not allow arbitrary bytes to be sent to hardware — use the command protocol only
  - Do not autodetect COM ports — require explicit configuration in config/devices.json
  - Do not allow hardware commands at autonomy_level < 2
  - Do not use serial.Serial in async code directly — always run_in_executor

PATHS:
  core/hardware/serial_controller.py — extend
  core/hardware/device_registry.py   — new
  core/tools/hardware_tools.py       — new

VERIFICATION:
  python -c "from core.hardware.device_registry import DeviceRegistry; print('OK')"
  python -c "from core.tools.hardware_tools import read_sensor; print('OK')"
SESSION 11 — MULTI-MODEL + MODEL SWITCHING + LOCAL FINE-TUNING PREP
Goal: Support multiple LLMs simultaneously — route tasks to the best model for each job.

Prompt to paste at session start:

You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-10 complete.

THIS SESSION GOAL: Multi-Model Intelligence — use different models for different task types.

MODEL ROUTING STRATEGY:
  deepseek-r1:8b    → reasoning, planning (slow but smart)
  mistral:7b        → fast chat responses, intent classification
  llava             → vision tasks (already works)
  llama3:8b         → general conversation, memory synthesis
  nomic-embed-text  → embeddings (replace all-MiniLM-L6-v2 for better quality)

FILES TO CREATE:
  core/llm/model_router.py — NEW:
    - ModelRouter class
    - route(task_type) → model_name
      task_type: "planning", "chat", "vision", "embedding", "synthesis"
    - register_model(task_type, model_name, priority=1)
    - is_available(model_name) → bool — check via Ollama /api/tags
    - get_fastest_available(task_type) → model_name — fallback chain
    - Config-driven: read model assignments from config/jarvis.ini [models] section
    - If preferred model offline: log warning, use fallback — never crash

  core/llm/model_health.py — NEW:
    - ModelHealthChecker class
    - check_all() → dict of {model_name: {"available": bool, "latency_ms": int}}
    - Runs on startup and every 60 seconds in background
    - Expose as /health/models in dashboard API (Session 9)

FILE MODIFICATIONS:
  core/llm/client.py:
    - Add model_router: ModelRouter = None to __init__
    - In complete(): if model_router: self.model = model_router.route(task_type)
    - Add task_type param to complete(), complete_json(), chat()

  core/llm/task_planner.py:
    - Use model_router.route("planning") for model selection

  core/memory/semantic_memory.py:
    - Add support for nomic-embed-text via Ollama embeddings API
    - If nomic-embed-text available: use it instead of sentence-transformers
    - Falls back to sentence-transformers if Ollama embedding not available

  config/jarvis.ini — add section:
    [models]
    planning_model = deepseek-r1:8b
    chat_model = mistral:7b
    vision_model = llava
    synthesis_model = llama3:8b
    embedding_model = nomic-embed-text
    fallback_model = mistral:7b

DO:
  - Model router must work even if only ONE model is installed
  - Never hardcode model names in code — always read from config or model_router
  - ModelHealthChecker must run as background asyncio task — never block startup
  - If all models offline: enter "offline mode" — use keyword responses only

DO NOT:
  - Do not call Ollama /api/tags on every request — cache results for 60 seconds
  - Do not switch models mid-conversation — route at conversation start
  - Do not require any new model to be installed — all should be optional

PATHS:
  core/llm/model_router.py       — new
  core/llm/model_health.py       — new
  core/llm/client.py             — modify
  core/llm/task_planner.py       — modify
  core/memory/semantic_memory.py — modify

VERIFICATION:
  python -c "from core.llm.model_router import ModelRouter; print('OK')"
  python -c "from core.llm.model_health import ModelHealthChecker; print('OK')"
SESSION 12 — SECURITY HARDENING + FULL TEST SUITE
Goal: Lock everything down. Every module has tests. No secrets leak. Jarvis is production-ready.

Prompt to paste at session start:

You are an expert Python systems engineer and security auditor.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-11 complete.

INSTALL BEFORE STARTING:
  pip install pytest pytest-asyncio pytest-cov bandit

THIS SESSION GOAL: Security + Testing — harden every layer and achieve 80%+ test cov