# JARVIS — COMPLETE CAPABILITY ROADMAP
# All Future Sessions — Full Instructions for Each

---

## MASTER CAPABILITY MAP

```
PHASE 5  (Next)   → Voice + Dynamic Integrations + Agent Upgrades
PHASE 6           → Coding Agent + File Editor + Sandboxed Execution
PHASE 7           → GUI Automation + Screen Understanding
PHASE 8           → Proactive Intelligence + Scheduling + Goals
PHASE 9           → User Profiling + Personalization + Synthesis
PHASE 10          → Dashboard + Web UI + Remote Control
PHASE 11          → Hardware / IoT + Serial Control
PHASE 12          → Multi-Model + Model Switching + Local Fine-tuning
PHASE 13          → Security Hardening + Full Test Suite
```

---
---

# SESSION 5 — VOICE + DYNAMIC INTEGRATIONS + AGENT UPGRADES

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis — local agentic AI at D:\AI\Jarvis\
Python 3.11, Windows 11, Ollama at http://localhost:11434
All code must be complete files — no snippets, no "add this here".

CURRENT WORKING STATE:
- main.py boots cleanly
- core/controller_v2.py — JarvisControllerV2 with start(), run_cli(), shutdown()
- core/memory/hybrid_memory.py — SQLite + ChromaDB working
- core/llm/task_planner.py — JSON planner with SYSTEM_TOOL_SCHEMA wired in
- core/agent/agent_loop.py — full plan→risk→confirm→execute→reflect loop
- core/autonomy/risk_evaluator.py — LOW/MEDIUM/HIGH/CRITICAL table-driven
- core/tools/tool_router.py + builtin_tools.py — tool registry working
- integrations/base.py (BaseIntegration ABC) + registry.py exist but are stubs
- core/voice/ files exist but nothing is wired

INSTALL BEFORE STARTING THIS SESSION:
  pip install httpx aiohttp faster-whisper pyttsx3 edge-tts pvporcupine

DELIVERABLE 1 — VOICE (files to create/replace):
  core/voice/stt.py          — Speech-to-text via faster-whisper
  core/voice/tts.py          — TTS: try edge-tts → pyttsx3 → print fallback
  core/voice/wake_word.py    — Wake word: pvporcupine → keyword fallback
  core/voice/voice_loop.py   — Async loop: wake→record→stt→process→tts
  core/controller_v2.py      — Add voice mode branch in run_cli()

  Voice rules:
  - python main.py --voice enables it, CLI is default
  - All voice deps wrapped in try/except ImportError — never crash on missing dep
  - STT fallback: if faster-whisper missing, read from input()
  - TTS fallback chain: edge-tts → pyttsx3 → print()
  - Wake word fallback: if pvporcupine missing, listen continuously
  - Never block the event loop — use asyncio.get_running_loop().run_in_executor()
  - Config in config/jarvis.ini under [voice]: enabled, stt_model, tts_engine, tts_voice, wake_word

DELIVERABLE 2 — DYNAMIC INTEGRATIONS (files to create/replace):
  integrations/base.py       — Updated ABC with is_available(), get_tools(), execute()
  integrations/registry.py   — register(), get_tools(), execute(), list_active()
  integrations/loader.py     — NEW: auto-discovers and loads clients/ subclasses
  integrations/clients/email.py      — send_email, read_emails, search_emails (smtplib+imaplib only)
  integrations/clients/whatsapp.py   — send_whatsapp (twilio, skip gracefully if not configured)
  integrations/clients/calendar.py   — add_event, list_events (icalendar, fallback to local .ics)
  integrations/clients/weather.py    — update existing to match new ABC

  Integration rules:
  - All secrets from os.environ or config/settings.env — never in code
  - Every execute() returns {"success": bool, "data": ..., "error": str|None}
  - integrations must NOT import from core.agent — one-way dependency only
  - loader.py scans integrations/clients/*.py, finds BaseIntegration subclasses
  - Only register if is_available() returns True — log skips with reason
  - Wire into core/agent/controller.py: call loader.load_all(config, registry) in __init__
  - Wire into core/execution/dispatcher.py: after TOOL_REGISTRY, check integration_registry
  - Wire into core/llm/task_planner.py: merge registry.get_tools() into SYSTEM_TOOL_SCHEMA

DELIVERABLE 3 — AGENT/LLM/MEMORY UPGRADES (modify only listed files):
  core/agent/agent_loop.py:
    - MAX_ITERATIONS: replace hardcoded 5 with configurable self.max_iterations (default 10)
    - REFLECT_SYSTEM_PROMPT: replace with expert software engineer persona (see below)
    - Add _truncate_observation(text, max_chars=800) helper — keep first/last 400 chars
    - Add think_blocks: list[str] to ExecutionTrace, save <think>...</think> before stripping
  core/llm/client.py:
    - Add _get_workspace_map(path, max_depth=3, max_files=50) — compact dir tree as string
    - Add workspace_path param to _build_system() — inject WORKSPACE: section if provided
    - Add keep_think=False param to complete() — preserve think blocks when True
  core/autonomy/risk_evaluator.py:
    - Add CONFIRM = 2 to RiskLevel (between MEDIUM and HIGH, shift HIGH to 3, CRITICAL to 4)
    - Add list_directory, read_file, search_code, run_linter to _DEFAULT_LOW
    - Move write_file, execute_shell, launch_application to _DEFAULT_CONFIRM (new set)
  core/memory/hybrid_memory.py:
    - Add store_code_file(file_path, content) — AST parse, store functions/classes as chunks
    - Add index_codebase(root_path) — hash files, only re-index changed ones, return stats
    - Wire index_codebase into initialize(index_path="") — call if index_path provided

NEW REFLECT_SYSTEM_PROMPT:
  "You are an expert software engineer and AI assistant named Jarvis.
   Review the executed plan and all tool observations.
   If any tool returned error or failure: state the technical root cause first,
   explain what went wrong, propose the specific fix.
   If successful: summarize what was done, show key output the user needs.
   Be direct and technical. No filler. Speak in second person."

FILE DELIVERY ORDER (must be this order to avoid import errors):
  1. integrations/base.py
  2. integrations/registry.py
  3. integrations/loader.py
  4. integrations/clients/email.py
  5. integrations/clients/whatsapp.py
  6. integrations/clients/calendar.py
  7. integrations/clients/weather.py
  8. core/autonomy/risk_evaluator.py
  9. core/memory/hybrid_memory.py
  10. core/llm/client.py
  11. core/agent/agent_loop.py
  12. core/voice/stt.py
  13. core/voice/tts.py
  14. core/voice/wake_word.py
  15. core/voice/voice_loop.py
  16. core/controller_v2.py
  17. core/execution/dispatcher.py
  18. core/agent/controller.py
  19. core/llm/task_planner.py

VERIFICATION after all files delivered:
  python -c "from integrations.loader import load_all; print('OK')"
  python -c "from core.agent.agent_loop import AgentLoopEngine; print('OK')"
  python -c "from core.memory.hybrid_memory import HybridMemory; print('OK')"
  python main.py --help
```

---
---

# SESSION 6 — CODING AGENT

**Goal:** Make Jarvis able to write, edit, run, and debug code autonomously.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis — local agentic AI at D:\AI\Jarvis\
Python 3.11, Windows 11, Ollama at http://localhost:11434

CURRENT STATE: Sessions 1-5 complete. main.py boots. Voice and integrations working.

INSTALL BEFORE STARTING:
  pip install docker pygments tree-sitter pyflakes

THIS SESSION GOAL: Build a Coding Agent — Jarvis can write, edit, run, and debug code.

FILES TO CREATE:
  core/tools/code_tools.py
    - edit_file(path, search_block, replace_block) — search-and-replace patch, NOT full overwrite
      Rules: read file, find search_block exactly once, replace with replace_block, write back
      If search_block not found: return error with diff showing current file content
      If found more than once: return error — ambiguous match
    - create_file(path, content) — create new file, fail if exists unless overwrite=True
    - read_file_with_lines(path) — return content with line numbers prepended
    - run_linter(path) — run pyflakes on a .py file, return structured output
    - search_code(root, pattern, file_glob="*.py") — grep-style search, return file:line:match
    - get_directory_tree(path, max_depth=3) — return compact tree string

  core/tools/sandbox.py
    - SandboxExecutor class
    - execute(command, working_dir, timeout=30) — run command in subprocess with timeout
    - If Docker available: run inside jarvis-sandbox container (python:3.11-slim)
    - If Docker NOT available: run in subprocess with os.chdir restriction + timeout kill
    - Always return {"stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}
    - NEVER allow commands: rm -rf, del /f, format, shutdown, reboot, pip install --system
    - Log every execution to audit logger before running

  core/agent/coding_loop.py
    - CodingLoopEngine class extending AgentLoopEngine
    - Override max_iterations to 15 by default
    - Add _run_tests(test_path) step after each code write
    - Add _auto_fix_linter(path) — run linter, if errors exist, ask LLM to fix, write, repeat (max 3x)
    - Add method save_think_blocks_to_file(trace, output_path) — write LLM reasoning to .md file

FILE MODIFICATIONS:
  core/autonomy/risk_evaluator.py — add to _DEFAULT_LOW:
    "edit_file", "create_file", "read_file_with_lines", "run_linter",
    "search_code", "get_directory_tree"
  core/autonomy/risk_evaluator.py — add to _DEFAULT_CONFIRM:
    "sandbox_execute", "run_tests"
  core/agent/controller.py — add coding mode:
    if intent == "coding" or any code keyword: use CodingLoopEngine instead of AgentLoopEngine
  core/llm/task_planner.py — add code tools to SYSTEM_TOOL_SCHEMA

DO:
  - edit_file must use search-and-replace blocks — never overwrite whole file
  - All sandbox execution goes through SandboxExecutor — never bare subprocess in tools
  - run_linter must return structured dict: {"errors": [...], "warnings": [...], "clean": bool}
  - search_code must respect .gitignore patterns — skip __pycache__, .git, venv
  - Save LLM <think> blocks to workspace/thinking_logs/ for every coding task

DO NOT:
  - Do not allow the agent to pip install packages without CONFIRM level permission
  - Do not allow the agent to modify files outside workspace/ and the project root
  - Do not run tests without first checking they exist
  - Do not use subprocess.shell=True — always use list form ["python", "file.py"]
  - Do not skip the linter step — every file write must be followed by run_linter

PATHS:
  core/tools/code_tools.py        — new file
  core/tools/sandbox.py           — new file
  core/agent/coding_loop.py       — new file
  core/autonomy/risk_evaluator.py — modify
  core/agent/controller.py        — modify (add coding mode branch)
  core/llm/task_planner.py        — modify (add code tool schemas)

VERIFICATION:
  python -c "from core.tools.code_tools import edit_file, run_linter; print('OK')"
  python -c "from core.tools.sandbox import SandboxExecutor; print('OK')"
  python -c "from core.agent.coding_loop import CodingLoopEngine; print('OK')"
  python main.py --help
```

---
---

# SESSION 7 — GUI AUTOMATION + SCREEN UNDERSTANDING

**Goal:** Jarvis can see your screen, click, type, and control desktop apps.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-6 complete.

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
```

---
---

# SESSION 8 — PROACTIVE INTELLIGENCE + SCHEDULER + GOALS

**Goal:** Jarvis runs in the background, pursues goals, sends reminders, and acts without being asked.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-7 complete.

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
```

---
---

# SESSION 9 — USER PROFILING + PERSONALIZATION + SYNTHESIS

**Goal:** Jarvis learns who you are, adapts its tone, and remembers your preferences across sessions.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-8 complete.

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
```

---
---

# SESSION 10 — DASHBOARD + WEB UI + REMOTE CONTROL

**Goal:** A local web dashboard to monitor Jarvis, view logs, manage memory, and send commands remotely.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-9 complete.

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
```

---
---

# SESSION 11 — HARDWARE / IoT + SERIAL CONTROL

**Goal:** Jarvis controls physical hardware — Arduino, sensors, lights, motors via serial.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-10 complete.

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
```

---
---

# SESSION 12 — MULTI-MODEL + MODEL SWITCHING + LOCAL FINE-TUNING PREP

**Goal:** Support multiple LLMs simultaneously — route tasks to the best model for each job.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-11 complete.

THIS SESSION GOAL: Multi-Model Intelligence — use different models for different task types.

MODEL ROUTING STRATEGY:
  deepseek-r1:8b    → reasoning, planning, code (slow but smart)
  mistral:7b        → fast chat responses, intent classification
  llava             → vision tasks (already works)
  llama3:8b         → general conversation, memory synthesis
  codellama:7b      → code generation specifically (if installed)
  nomic-embed-text  → embeddings (replace all-MiniLM-L6-v2 for better quality)

FILES TO CREATE:
  core/llm/model_router.py — NEW:
    - ModelRouter class
    - route(task_type) → model_name
      task_type: "planning", "coding", "chat", "vision", "embedding", "synthesis"
    - register_model(task_type, model_name, priority=1)
    - is_available(model_name) → bool — check via Ollama /api/tags
    - get_fastest_available(task_type) → model_name — fallback chain
    - Config-driven: read model assignments from config/jarvis.ini [models] section
    - If preferred model offline: log warning, use fallback — never crash

  core/llm/model_health.py — NEW:
    - ModelHealthChecker class
    - check_all() → dict of {model_name: {"available": bool, "latency_ms": int}}
    - Runs on startup and every 60 seconds in background
    - Expose as /health/models in dashboard API (Session 10)

FILE MODIFICATIONS:
  core/llm/client.py:
    - Add model_router: ModelRouter = None to __init__
    - In complete(): if model_router: self.model = model_router.route(task_type)
    - Add task_type param to complete(), complete_json(), chat()

  core/llm/task_planner.py:
    - Use model_router.route("planning") for model selection
    - Use model_router.route("coding") if intent is coding

  core/memory/semantic_memory.py:
    - Add support for nomic-embed-text via Ollama embeddings API
    - If nomic-embed-text available: use it instead of sentence-transformers
    - Falls back to sentence-transformers if Ollama embedding not available

  config/jarvis.ini — add section:
    [models]
    planning_model = deepseek-r1:8b
    chat_model = mistral:7b
    coding_model = deepseek-r1:8b
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
```

---
---

# SESSION 13 — SECURITY HARDENING + FULL TEST SUITE

**Goal:** Lock everything down. Every module has tests. No secrets leak. Jarvis is production-ready.

**Prompt to paste at session start:**

```
You are an expert Python systems engineer and security auditor.
Project: Jarvis at D:\AI\Jarvis\. Sessions 1-12 complete.

INSTALL BEFORE STARTING:
  pip install pytest pytest-asyncio pytest-cov bandit

THIS SESSION GOAL: Security + Testing — harden every layer and achieve 80%+ test coverage.

SECURITY AUDIT — fix these in the existing code:

  1. core/tools/builtin_tools.py:
     - ALLOWED_DIRECTORIES only has workspace/ and outputs/ — verify this is enforced everywhere
     - Add path traversal check: reject any path containing ".." or symlinks outside sandbox
     - Add max file size check for read_file: reject files > 10MB

  2. core/execution/dispatcher.py:
     - Add input sanitization before any tool call: strip null bytes, limit arg string length to 4096 chars
     - Rate limiting: max 30 tool calls per minute per session — return error if exceeded

  3. core/logger.py:
     - Audit log is already append-only + hash-chained — verify chain integrity on load
     - Add log rotation: if audit.jsonl > 50MB, archive and start fresh

  4. integrations/ (all clients):
     - Verify no API keys are logged — scrub secrets from all log messages
     - Add request timeout to every external API call (default 10s, configurable)
     - Add retry logic: max 2 retries with 2s backoff for network errors

  5. dashboard/server.py:
     - Verify auth token is checked on all POST endpoints
     - Add rate limiting: max 60 requests/minute per IP
     - Add Content-Security-Policy header to all responses

TEST FILES TO CREATE:
  tests/test_risk_evaluator.py    — test all risk levels, all tool names, edge cases
  tests/test_memory.py            — test SQLite store/recall, AST chunking, hash-based reindex
  tests/test_agent_loop.py        — test full plan→execute→reflect with mock tools
  tests/test_tools.py             — test edit_file search-replace, sandbox, linter
  tests/test_integrations.py      — test each integration's is_available() and mock execute()
  tests/test_profile.py           — test profile save/load, synthesis delta merge
  tests/test_dispatcher.py        — test routing, rate limiting, input sanitization
  tests/test_voice.py             — test STT/TTS fallback chains (mock audio)
  tests/test_security.py          — test path traversal blocked, secrets not logged, rate limits

TEST RULES:
  - All async tests use @pytest.mark.asyncio
  - All external calls (Ollama, APIs) must be mocked — tests must pass with no internet/Ollama
  - Use pytest fixtures for: temp_dir, mock_config, mock_llm, mock_memory
  - Every test file must have at least 8 test cases
  - Target: pytest --cov=core --cov-report=term-missing should show >80% coverage

DO:
  - Run bandit -r core/ after all fixes — target: no HIGH severity findings
  - Add pre-commit safety check: before any file write, confirm path is in sandbox
  - Document every security decision in a SECURITY.md file in project root

DO NOT:
  - Do not use unittest — use pytest only
  - Do not mock the filesystem — use pytest's tmp_path fixture for real temp dirs
  - Do not skip testing error paths — unhappy paths are more important than happy paths
  - Do not add security that breaks existing functionality — test both before and after

PATHS:
  tests/test_risk_evaluator.py  — new
  tests/test_memory.py          — new
  tests/test_agent_loop.py      — new
  tests/test_tools.py           — new
  tests/test_integrations.py    — new
  tests/test_profile.py         — new
  tests/test_dispatcher.py      — new
  tests/test_voice.py           — new
  tests/test_security.py        — new
  SECURITY.md                   — new

VERIFICATION:
  pip install pytest pytest-asyncio pytest-cov bandit
  pytest tests/ -v --tb=short
  bandit -r core/ -ll
```

---
---

## QUICK REFERENCE — WHAT EACH SESSION ADDS

| Session | What You Get After It |
|---------|----------------------|
| 5 | Voice (speak to Jarvis), Email/WhatsApp/Calendar integrations |
| 6 | Jarvis writes and debugs code, runs linter, uses search-replace edits |
| 7 | Jarvis sees your screen, clicks, types, controls apps |
| 8 | Jarvis reminds you of goals, runs scheduled tasks, monitors system |
| 9 | Jarvis knows your name, style, preferences — adapts automatically |
| 10 | Web dashboard at localhost:7070 — monitor and control from browser |
| 11 | Jarvis controls Arduino/hardware via serial port |
| 12 | Multiple LLMs for different tasks — faster chat, smarter planning |
| 13 | Security hardened, 80%+ test coverage, production ready |

## RECOMMENDED ORDER

Do sessions in order. Each one builds on the last.
Sessions 6 and 7 can be swapped if you prefer GUI before coding.
Sessions 11 and 12 are independent — can be done in either order.
Session 13 should always be last.