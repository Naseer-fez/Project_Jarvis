# JARVIS — COMBINED SESSION ROADMAP
# Maximum capability, minimum sessions
# Based on actual codebase analysis — no invented files

---

## WHAT ALREADY EXISTS (DO NOT REBUILD)
- core/voice/stt.py            — SpeechToText class, Whisper, PyAudio ✅
- core/voice/tts.py            — TextToSpeech class, Piper TTS ✅
- core/voice/wake_word.py      — WakeWordDetector, Porcupine ✅
- core/voice/voice_layer.py    — VoiceLayer V2 full implementation ✅
- core/voice/voice_loop.py     — async loop orchestrator ✅
- core/autonomy/goal_manager.py — Goal dataclass + GoalManager ✅
- core/agentic/scheduler.py    — ScheduledMission + pull-based scheduler ✅
- core/agentic/belief_state.py — Belief + BeliefState ✅
- core/agentic/mission.py      — Mission + MissionStep ✅
- core/metrics/confidence.py   — ConfidenceModel weighted scorer ✅
- core/introspection/health.py — HealthCheck + HealthReport ✅
- core/profile.py              — UserProfileEngine (partial) ✅
- core/synthesis.py            — ProfileSynthesizer (partial) ✅
- integrations/base.py         — BaseIntegration ABC + ToolResult ✅
- integrations/registry.py     — api_registry with register/get_tool/list_schemas ✅

## WHAT IS MISSING (needs to be built)
- Nothing is wired together — voice, goals, scheduler all exist but nothing calls them
- No integration clients (email, whatsapp, calendar)
- No web dashboard
- No proactive notifications
- No multi-model routing
- No hardware tools
- No tests

---
---

# SESSION A — WIRE EVERYTHING + VOICE + INTEGRATIONS
# (Combines old Sessions 5 + parts of 8 + 9)

**What this delivers:**
- Jarvis speaks and listens (voice files already exist — just wire them)
- Goals and scheduler work end-to-end (data classes exist — just wire them)
- Email, WhatsApp, Calendar integrations (dynamic plugin loader)
- User profile learns and adapts across sessions
- Confidence model gates autonomous actions
- Health monitoring runs on startup

---

**Paste this entire block at the start of your session:**

```
You are an expert Python systems engineer.
Project: Jarvis — local agentic AI at D:\AI\Jarvis\
Python 3.11+, Windows 11, Ollama at http://localhost:11434
Deliver COMPLETE files only — no partial snippets, no "add this here".

════════════════════════════════════════════════
CRITICAL: READ THESE FILES FIRST BEFORE WRITING ANYTHING
════════════════════════════════════════════════
Read every file listed below completely before writing a single line.
The files already contain substantial code. Your job is to wire them together,
not rewrite them.

Files to read first:
  core/voice/voice_layer.py       (585 lines — full VoiceLayer V2)
  core/voice/voice_loop.py        (361 lines — async loop)
  core/voice/stt.py               (197 lines — SpeechToText)
  core/voice/tts.py               (181 lines — TextToSpeech)
  core/voice/wake_word.py         (159 lines — WakeWordDetector)
  core/autonomy/goal_manager.py   (199 lines — GoalManager)
  core/agentic/scheduler.py       (213 lines — pull-based scheduler)
  core/metrics/confidence.py      (140 lines — ConfidenceModel)
  core/introspection/health.py    (180 lines — HealthReport)
  core/profile.py                 (112 lines — UserProfileEngine)
  core/synthesis.py               (93 lines  — ProfileSynthesizer)
  integrations/base.py            (135 lines — BaseIntegration ABC)
  integrations/registry.py        (141 lines — api_registry)
  core/agent/controller.py        (328 lines — MainController)
  core/controller_v2.py           (103 lines — JarvisControllerV2)
  main.py                         (318 lines — entry point)

════════════════════════════════════════════════
INSTALL BEFORE THIS SESSION
════════════════════════════════════════════════
pip install httpx aiohttp pvporcupine pvrecorder faster-whisper
pip install pyttsx3 sounddevice soundfile numpy
pip install apscheduler plyer

════════════════════════════════════════════════
PART 1 — WIRE VOICE INTO CONTROLLER
════════════════════════════════════════════════

FILE: core/controller_v2.py  (MODIFY — extend existing JarvisControllerV2)

Add to __init__(self, config=None, voice=False, ...):
  self.voice_enabled = voice
  self._voice_layer = None
  if voice:
      try:
          from core.voice.voice_layer import VoiceLayer
          self._voice_layer = VoiceLayer(
              config=config,
              text_handler=self._voice_text_handler
          )
      except ImportError as e:
          logging.getLogger(__name__).warning(f"Voice unavailable: {e}")

Add method async _voice_text_handler(self, text: str) -> str:
  — calls self.process(text) and returns the string response

Modify async run_cli(self):
  — if self._voice_layer is not None:
      await self._voice_layer.start()
      await self._voice_layer.stop()  # on shutdown
  — else: keep existing CLI input loop exactly as is

Add to async shutdown(self):
  — if self._voice_layer: await self._voice_layer.stop()

RULES:
- Do NOT rewrite VoiceLayer — it already exists and works
- voice_layer.start() is already async — call it directly with await
- If VoiceLayer import fails: log warning and continue in CLI mode
- Keep all existing JarvisControllerV2 methods intact

════════════════════════════════════════════════
PART 2 — WIRE GOALS + SCHEDULER INTO CONTROLLER
════════════════════════════════════════════════

FILE: core/controller_v2.py  (same file, continuing)

Add to __init__:
  from core.autonomy.goal_manager import GoalManager
  from core.agentic.scheduler import Scheduler  # read actual class name from file
  self.goal_manager = GoalManager()
  self.scheduler = Scheduler()  # use actual class name from scheduler.py

Add to process(self, user_input: str) -> str:
  — detect goal intent: if any of ("remind me", "set goal", "schedule", "don't forget")
    in user_input.lower():
      → parse the goal from the text
      → call self.goal_manager.create(description=...) or equivalent method
      → return "✓ Goal set: {description}"
  — detect goal query: if "what are my goals" or "show goals" in user_input.lower():
      → return formatted list of active goals

Add background check method async _check_due_goals(self):
  — every 5 minutes: call scheduler.due() to get overdue items
  — for each due item: print notification to terminal
  — if voice enabled: call self._voice_layer.speak(notification_text)

Wire _check_due_goals into run_cli() as a background asyncio task

RULES:
- Read goal_manager.py and scheduler.py first — use the ACTUAL method names
- Persist goals to memory/goals.json — load on GoalManager init
- Do not crash if goals.json doesn't exist yet

════════════════════════════════════════════════
PART 3 — WIRE CONFIDENCE MODEL INTO AGENT LOOP
════════════════════════════════════════════════

FILE: core/agent/agent_loop.py  (MODIFY)

Add to AgentLoopEngine.__init__:
  from core.metrics.confidence import ConfidenceModel
  self.confidence = ConfidenceModel()

After IntentClassifierV2 returns confidence score in run():
  self.confidence.update("intent_clarity", classification_confidence)

After each ToolObservation in run():
  success_score = 1.0 if obs.execution_status == "success" else 0.0
  self.confidence.update("tool_reliability", success_score)

Before executing any CONFIRM-level tool:
  score = self.confidence.score()
  if score < 0.4:  # low confidence — always ask user
      force_confirmation = True

Add think_blocks: list[str] = field(default_factory=list) to ExecutionTrace
In _reflect(): extract <think>...</think> with regex, save to trace.think_blocks

Replace MAX_ITERATIONS = 5 with self.max_iterations = 10 (configurable)

Replace REFLECT_SYSTEM_PROMPT with:
  "You are Jarvis, an expert AI assistant. Review the executed plan and observations.
   If any tool failed: state the root cause first, then the fix.
   If successful: summarize concisely what was accomplished.
   Be direct, technical, no filler phrases. Speak to the user in second person."

Add truncation helper before class:
  def _truncate_obs(text: str, max_chars: int = 800) -> str:
      if len(text) <= max_chars: return text
      h = max_chars // 2
      return text[:h] + f"\n...[{len(text)-max_chars} chars omitted]...\n" + text[-h:]

Use _truncate_obs() on every obs.output_summary in _reflect()

════════════════════════════════════════════════
PART 4 — WIRE USER PROFILE + SYNTHESIS
════════════════════════════════════════════════

FILE: core/profile.py  (REPLACE — keep class name UserProfileEngine)

Fields to persist in memory/user_profile.json:
  name, communication_style (formal/casual/technical),
  expertise_level (beginner/intermediate/advanced/expert),
  preferred_topics: list[str], timezone, language,
  interaction_count, first_seen (ISO string), last_seen (ISO string)

Methods required:
  __init__() — load from memory/user_profile.json, use defaults if missing
  save() — atomic write (write to .tmp then rename to avoid corruption)
  update_from_conversation(user_text, jarvis_response) — increment interaction_count,
    update last_seen, detect name if "my name is X" in text
  get_system_prompt_injection() -> str — max 150 tokens:
    "User: {name}. Style: {communication_style}. Level: {expertise_level}."
  get_communication_style() -> str:
    formal → "Be precise and professional."
    casual → "Be friendly and conversational."
    technical → "Be detailed and technical. Use correct terminology."
    default → "Be helpful and clear."

FILE: core/synthesis.py  (REPLACE — keep class name ProfileSynthesizer)

Methods required:
  __init__(llm) — store llm reference
  async synthesize(recent_conversations: list[str], profile: UserProfileEngine) -> dict:
    — build prompt from last 20 conversation strings
    — call LLM with SYNTHESIS_SYSTEM prompt
    — parse JSON delta: only update fields with confidence > 0.6
    — call profile.save() after merge
    — return {"updated_fields": [...], "delta": {...}}
  should_run(profile) -> bool:
    — return True if interaction_count % 20 == 0 and interaction_count > 0

FILE: core/controller_v2.py  (same file, add profile wiring)

Add to __init__:
  from core.profile import UserProfileEngine
  from core.synthesis import ProfileSynthesizer
  self.profile = UserProfileEngine()
  self.synthesizer = ProfileSynthesizer(self.llm)
  self._conversation_buffer: list[str] = []

After every self.process() call returns a response:
  self.profile.update_from_conversation(user_input, response)
  self._conversation_buffer.append(f"User: {user_input}\nJarvis: {response}")
  if self.synthesizer.should_run(self.profile):
      asyncio.create_task(
          self.synthesizer.synthesize(self._conversation_buffer[-20:], self.profile)
      )
      self._conversation_buffer.clear()

In _build_system / context injection (wherever LLM prompt is built):
  inject self.profile.get_system_prompt_injection() + self.profile.get_communication_style()

RULES for profile:
  - Atomic writes only — never corrupt on crash
  - Human-readable JSON — user may hand-edit it
  - Never store passwords, health data, or financial info
  - Synthesis runs as background task — never block user interaction

════════════════════════════════════════════════
PART 5 — DYNAMIC INTEGRATION FRAMEWORK
════════════════════════════════════════════════

FILE: integrations/loader.py  (NEW)

class IntegrationLoader:
  def load_all(config, registry) -> dict:  (returns {"loaded": [...], "skipped": [...]})
    — scan integrations/clients/*.py
    — import each file
    — find all subclasses of BaseIntegration in the module
    — call integration.is_available()
    — if True: call registry.register(integration) or equivalent
    — if False: log warning with reason, add to skipped list
    — catch ALL exceptions per plugin — one bad plugin must not block others

FILE: integrations/base.py  (MODIFY — keep existing, add is_available())

Add to BaseIntegration ABC:
  name: str = ""           — unique slug e.g. "email"
  description: str = ""   — human-readable
  required_config: list[str] = []  — env var names that must be set

  @abstractmethod
  def is_available(self) -> bool:
    — check if required deps installed AND required_config vars are set
    — return False (not raise) if missing

  @abstractmethod
  def get_tools(self) -> list[dict]:
    — return list of tool defs in SYSTEM_TOOL_SCHEMA format

Keep existing: tool_name, risk_level, tool_schema, execute(), ToolResult

FILE: integrations/clients/email.py  (NEW)

class EmailIntegration(BaseIntegration):
  name = "email"
  required_config = ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "SMTP_HOST", "IMAP_HOST"]
  
  is_available(): check smtplib/imaplib importable (stdlib) AND env vars set
  
  Tools: send_email(to, subject, body), read_emails(folder, limit=10), search_emails(query)
  
  All use smtplib + imaplib only — no external packages
  Credentials from os.environ only — never hardcoded
  Every execute() returns {"success": bool, "data": ..., "error": str|None}
  Timeout 10s on all connections

FILE: integrations/clients/whatsapp.py  (NEW)

class WhatsAppIntegration(BaseIntegration):
  name = "whatsapp"
  required_config = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]
  
  is_available(): check twilio importable AND env vars set
  
  Tools: send_whatsapp(to, message)
  All wrapped in try/except — return error ToolResult on any failure

FILE: integrations/clients/calendar.py  (NEW)

class CalendarIntegration(BaseIntegration):
  name = "calendar"
  required_config = []  — no required config (uses local .ics file)
  
  is_available(): always True (uses stdlib only, local file fallback)
  
  Tools: add_event(title, date, time, duration_minutes),
         list_events(days_ahead=7),
         search_events(query)
  
  Storage: memory/calendar.ics — create if not exists
  Use icalendar library if installed, else write raw iCal format manually

Wire integrations into existing system:
  core/agent/controller.py __init__:
    from integrations.loader import IntegrationLoader
    self.integration_loader = IntegrationLoader()
    self.integration_result = self.integration_loader.load_all(config, api_registry)
    logger.info(f"Integrations: {self.integration_result}")

  core/execution/dispatcher.py execute():
    After checking TOOL_REGISTRY — check api_registry.get_tool(tool_name)
    If found: return await integration.execute(**args)

  core/llm/task_planner.py _call_ollama():
    from integrations.registry import list_schemas
    integration_tools = list_schemas()
    merged_schema = {**SYSTEM_TOOL_SCHEMA,
                     "tools": SYSTEM_TOOL_SCHEMA["tools"] + integration_tools}
    — use merged_schema in the prompt

config/settings.env  (NEW FILE — template, not committed to git):
  EMAIL_ADDRESS=
  EMAIL_PASSWORD=
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  IMAP_HOST=imap.gmail.com
  TWILIO_ACCOUNT_SID=
  TWILIO_AUTH_TOKEN=
  TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

════════════════════════════════════════════════
PART 6 — HEALTH CHECK ON STARTUP
════════════════════════════════════════════════

FILE: core/introspection/health.py  (MODIFY — keep existing classes, add wire method)

Add function run_startup_health_check(controller) -> HealthReport:
  Checks to run:
  - ollama_reachable: GET http://localhost:11434 → ok/fail
  - chromadb_ready: try importing chromadb → ok/warn
  - memory_sqlite: check memory/memory.db accessible → ok/fail
  - voice_deps: check pvporcupine, sounddevice importable → ok/warn
  - config_loaded: check config/jarvis.ini exists → ok/fail
  - integrations_loaded: count how many loaded successfully → ok/warn

Print report to terminal on startup with colored output (use colorama if available)

FILE: main.py  (MODIFY — add startup health check)

After controller.start():
  from core.introspection.health import run_startup_health_check
  report = run_startup_health_check(controller)
  print(report.summary())  — use existing summary() method

════════════════════════════════════════════════
DO — RULES FOR THIS SESSION
════════════════════════════════════════════════
- Read every listed file fully before writing any code
- Use ACTUAL method names from existing files — do not invent new ones
- Wrap ALL optional dependency imports in try/except ImportError
- All async methods use asyncio.get_running_loop() not get_event_loop()
- All file writes use atomic pattern: write .tmp then os.replace()
- Secrets only from os.environ.get() or python-dotenv — never hardcoded
- Every integration's is_available() returns False silently — never raises
- Log every integration load attempt: success AND skip with reason
- Profile injection into LLM prompt must be under 150 tokens
- Synthesis runs as asyncio.create_task() — never awaited in main flow

════════════════════════════════════════════════
DO NOT — HARD RULES
════════════════════════════════════════════════
- Do NOT rewrite voice_layer.py — it is already complete
- Do NOT rewrite voice_loop.py — it is already complete
- Do NOT rewrite stt.py, tts.py, wake_word.py — they are already complete
- Do NOT rewrite goal_manager.py or scheduler.py data structures — extend only
- Do NOT create new data classes that duplicate existing ones
- Do NOT use asyncio.get_event_loop() (deprecated) anywhere
- Do NOT store any credential in any .py file or .json file
- Do NOT let one failed integration crash the loader — catch per-plugin
- Do NOT make integrations import from core.agent — one-way dependency only
- Do NOT block the event loop during synthesis — background task only
- Do NOT call synthesizer more than once per 20 interactions

════════════════════════════════════════════════
FILE DELIVERY ORDER
════════════════════════════════════════════════
Deliver in this exact order — each file must compile before next starts:

 1. config/settings.env              (template, new)
 2. integrations/base.py             (modify — add is_available, get_tools)
 3. integrations/loader.py           (new)
 4. integrations/clients/__init__.py (new, empty)
 5. integrations/clients/email.py    (new)
 6. integrations/clients/whatsapp.py (new)
 7. integrations/clients/calendar.py (new)
 8. core/profile.py                  (replace — keep UserProfileEngine)
 9. core/synthesis.py                (replace — keep ProfileSynthesizer)
10. core/introspection/health.py     (modify — add run_startup_health_check)
11. core/agent/agent_loop.py         (modify — confidence, think_blocks, truncate, prompt)
12. core/controller_v2.py            (modify — wire voice, goals, profile, scheduler)
13. core/execution/dispatcher.py     (modify — add integration routing)
14. core/agent/controller.py         (modify — add loader.load_all)
15. core/llm/task_planner.py         (modify — merge integration tools into prompt)
16. main.py                          (modify — add health check on startup)

════════════════════════════════════════════════
VERIFICATION — run after every file delivered
════════════════════════════════════════════════
python -c "from integrations.loader import IntegrationLoader; print('OK')"
python -c "from integrations.clients.email import EmailIntegration; print('OK')"
python -c "from core.profile import UserProfileEngine; p=UserProfileEngine(); print(p.get_communication_style())"
python -c "from core.agent.agent_loop import AgentLoopEngine, ExecutionTrace; print('OK')"
python main.py --help
python main.py --voice --help
```

---
---

# SESSION B — DASHBOARD + PROACTIVE MONITOR + MULTI-MODEL
# (Combines old Sessions 10 + 8 proactive + 12)

**What this delivers:**
- Local web dashboard at localhost:7070
- System resource monitor (CPU/RAM alerts)
- File change monitor
- Windows desktop notifications
- Multiple LLMs routed by task type

---

**Paste this entire block at the start of your session:**

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\. Session A is complete.
Python 3.11+, Windows 11, Ollama at http://localhost:11434

════════════════════════════════════════════════
INSTALL BEFORE THIS SESSION
════════════════════════════════════════════════
pip install fastapi uvicorn websockets jinja2 python-multipart plyer

════════════════════════════════════════════════
PART 1 — MULTI-MODEL ROUTING
════════════════════════════════════════════════

FILE: core/llm/model_router.py  (NEW)

class ModelRouter:
  Model assignments (all configurable via config/jarvis.ini [models]):
    planning   → deepseek-r1:8b
    chat       → mistral:7b
    vision     → llava
    synthesis  → llama3:8b (fallback: deepseek-r1:8b)
    embedding  → nomic-embed-text (fallback: all-MiniLM-L6-v2)
    fallback   → mistral:7b

  Methods:
    __init__(config=None) — load assignments from config or use defaults
    route(task_type: str) -> str — return model name for task
    is_available(model_name: str) -> bool:
      — GET http://localhost:11434/api/tags
      — cache result for 60 seconds — do NOT call on every request
      — return False if Ollama unreachable
    get_best_available(task_type: str) -> str:
      — try preferred model → if unavailable try fallback → return fallback always
    list_available() -> dict[str, bool]:
      — return all known models and their availability

FILE: core/llm/client.py  (MODIFY — keep all existing code)

Add to __init__(self, ...):
  self.model_router: ModelRouter | None = None

Add method set_router(router: ModelRouter):
  self.model_router = router

Modify complete(self, prompt, system="", temperature=0.1, task_type="chat") -> str:
  if self.model_router:
      model = self.model_router.get_best_available(task_type)
  else:
      model = self.model
  — use model in the payload instead of self.model
  — keep all existing error handling exactly as is

Modify complete_json(..., task_type="planning"):
  — pass task_type down to complete()

FILE: core/agent/controller.py  (MODIFY — MainController)

Add to __init__:
  from core.llm.model_router import ModelRouter
  self.model_router = ModelRouter(config=config)
  self.llm.set_router(self.model_router)
  self.planner_llm_type = "planning"  — used when calling planner

In _run_agent_task(): pass task_type="planning" when calling planner
In _simple_chat(): pass task_type="chat"

config/jarvis.ini — add section (LLM will produce this):
  [models]
  planning_model = deepseek-r1:8b
  chat_model = mistral:7b
  vision_model = llava
  synthesis_model = deepseek-r1:8b
  fallback_model = mistral:7b

RULES:
  - Model availability is cached 60s — never check on every request
  - If ALL models unavailable: return "Jarvis is offline — Ollama not running."
  - Never hardcode model names in logic — always use model_router.route()
  - ModelRouter must work if only ONE model is installed

════════════════════════════════════════════════
PART 2 — PROACTIVE MONITOR + NOTIFICATIONS
════════════════════════════════════════════════

FILE: core/proactive/__init__.py  (NEW — empty)

FILE: core/proactive/notifier.py  (NEW)

class NotificationManager:
  notify(message: str, level: str = "info") -> None:
    — always print to terminal with timestamp
    — try Windows toast: plyer.notification.notify(title="Jarvis", message=..., timeout=5)
    — if plyer not available: terminal only
    — if voice_layer available: speak the message (passed in as optional arg)

  schedule_reminder(message: str, in_seconds: int) -> None:
    — create asyncio.create_task(self._delayed_notify(message, in_seconds))

  async _delayed_notify(message: str, delay: int) -> None:
    — await asyncio.sleep(delay)
    — self.notify(message)

FILE: core/proactive/background_monitor.py  (NEW)

class BackgroundMonitor:
  __init__(notifier: NotificationManager, config=None):
    self.notifier = notifier
    self.cpu_threshold = 90   # from config [proactive] cpu_alert_threshold
    self.ram_threshold = 90
    self._tasks: list[asyncio.Task] = []
    self._running = False

  async start() -> None:
    self._running = True
    self._tasks.append(asyncio.create_task(self._monitor_resources()))

  async stop() -> None:
    self._running = False
    for t in self._tasks: t.cancel()

  async _monitor_resources() -> None:
    while self._running:
        await asyncio.sleep(60)  # check every 60 seconds
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            if cpu > self.cpu_threshold:
                self.notifier.notify(f"⚠️ CPU at {cpu:.0f}%", level="warn")
            if ram > self.ram_threshold:
                self.notifier.notify(f"⚠️ RAM at {ram:.0f}%", level="warn")
        except ImportError:
            pass  # psutil not installed — silently skip

Wire BackgroundMonitor into core/controller_v2.py:
  Add to __init__:
    from core.proactive.notifier import NotificationManager
    from core.proactive.background_monitor import BackgroundMonitor
    self.notifier = NotificationManager()
    self.monitor = BackgroundMonitor(self.notifier, config)

  Add to start():
    await self.monitor.start()

  Add to shutdown():
    await self.monitor.stop()

config/jarvis.ini — add section:
  [proactive]
  cpu_alert_threshold = 90
  ram_alert_threshold = 90
  goal_check_interval_minutes = 5

════════════════════════════════════════════════
PART 3 — WEB DASHBOARD
════════════════════════════════════════════════

FILES TO CREATE:
  dashboard/__init__.py    (empty)
  dashboard/server.py      (FastAPI app — all logic here)
  dashboard/templates/index.html    (main page — Jinja2)
  dashboard/templates/memory.html   (memory browser)
  dashboard/templates/goals.html    (goals view)
  dashboard/static/style.css        (dark theme)
  dashboard/static/app.js           (WebSocket client)

dashboard/server.py SPEC:

  Framework: FastAPI + Jinja2. No React, no npm, no build tools.
  Port: 7070 (configurable in jarvis.ini [dashboard] port)
  Bind: 127.0.0.1 ONLY — never 0.0.0.0

  State holder (module-level):
    _controller: JarvisControllerV2 | None = None
    def set_controller(c): global _controller; _controller = c

  Routes:
    GET  /           → index.html — session_id, state, last_response
    GET  /memory     → memory.html — search SQLite memory (query param: ?q=)
    GET  /goals      → goals.html — list active goals from goal_manager
    GET  /health     → JSON: {ollama: bool, memory: bool, voice: bool, integrations: [str]}
    POST /command    → accept {"text": "..."}, call controller.process(text), return response
                       Requires header: X-Dashboard-Token matching jarvis.ini [dashboard] secret
    POST /goals/add  → accept {"description": "...", "priority": 1}
    WS   /ws         → push every 2s: {state, session_id, memory_count, active_goals,
                         ollama_online, model, last_response}

  index.html MUST contain:
    - Current state badge (IDLE/THINKING/ACTING etc) — auto-updates via WebSocket
    - Last Jarvis response
    - Command input box + send button (POST /command)
    - Link to /memory and /goals
    - Session info: session_id, model, uptime

  memory.html MUST contain:
    - Search box (GET /memory?q=...)
    - Table of results: timestamp | category | content
    - Reads directly from memory/memory.db SQLite (no ORM)

  goals.html MUST contain:
    - List of active goals with priority and created_at
    - Add goal form
    - Complete/cancel buttons (POST to /goals/complete/{id})

  style.css: dark background (#1a1a2e), accent color (#4ecca3), monospace font

  security:
    - POST /command and POST /goals/* require X-Dashboard-Token header
    - Token read from jarvis.ini [dashboard] secret = change_this_token
    - If token missing or wrong: return 401

FILE: main.py  (MODIFY — add --dashboard flag)

Add argument: --dashboard (store_true)

If --dashboard:
  import threading, uvicorn
  from dashboard.server import app, set_controller
  set_controller(controller)
  def _run_dashboard():
      uvicorn.run(app, host="127.0.0.1", port=7070, log_level="warning")
  t = threading.Thread(target=_run_dashboard, daemon=True)
  t.start()
  print("Dashboard: http://localhost:7070")

config/jarvis.ini — add section:
  [dashboard]
  port = 7070
  secret = change_this_token

════════════════════════════════════════════════
DO — RULES FOR THIS SESSION
════════════════════════════════════════════════
- Dashboard binds 127.0.0.1 ONLY — this is a hard security requirement
- Model availability cache expires every 60s — no more frequent API calls
- Background monitor uses asyncio.sleep — no threads, no watchdog
- All dashboard POST routes validate token before doing anything
- Dashboard runs in daemon thread — main loop must still work if dashboard fails
- plyer import wrapped in try/except — notifications are optional
- psutil import wrapped in try/except — resource monitor is optional

════════════════════════════════════════════════
DO NOT
════════════════════════════════════════════════
- Do NOT use React, Vue, npm, webpack, or any frontend build tool
- Do NOT expose filesystem browsing via any API route
- Do NOT allow dashboard to trigger delete or high-risk tool calls
- Do NOT use SQLAlchemy — read SQLite directly with sqlite3 (stdlib)
- Do NOT use asyncio.get_event_loop() — use get_running_loop() inside async
- Do NOT let dashboard crash main.py — daemon=True + try/except in thread

════════════════════════════════════════════════
FILE DELIVERY ORDER
════════════════════════════════════════════════
 1. core/llm/model_router.py              (new)
 2. core/llm/client.py                    (modify — add set_router, task_type)
 3. core/proactive/__init__.py            (new, empty)
 4. core/proactive/notifier.py            (new)
 5. core/proactive/background_monitor.py  (new)
 6. core/controller_v2.py                 (modify — wire monitor + notifier)
 7. core/agent/controller.py              (modify — wire model_router)
 8. dashboard/__init__.py                 (new, empty)
 9. dashboard/server.py                   (new)
10. dashboard/static/style.css            (new)
11. dashboard/static/app.js               (new)
12. dashboard/templates/index.html        (new)
13. dashboard/templates/memory.html       (new)
14. dashboard/templates/goals.html        (new)
15. main.py                               (modify — add --dashboard flag)
16. config/jarvis.ini                     (show complete updated file)

════════════════════════════════════════════════
VERIFICATION
════════════════════════════════════════════════
python -c "from core.llm.model_router import ModelRouter; print('OK')"
python -c "from core.proactive.notifier import NotificationManager; print('OK')"
python -c "from dashboard.server import app; print('OK')"
python main.py --help     # must show --dashboard flag
python main.py --dashboard &
curl http://localhost:7070/health
```

---
---

# SESSION C — HARDWARE + GUI AUTOMATION + SECURITY + TESTS
# (Combines old Sessions 11 + 7 + 13)

**What this delivers:**
- Arduino/serial hardware control (existing SerialController extended)
- Screen capture + OCR + GUI click/type automation
- Security audit + path traversal prevention
- Full pytest test suite

---

**Paste this entire block at the start of your session:**

```
You are an expert Python systems engineer and security auditor.
Project: Jarvis at D:\AI\Jarvis\. Sessions A and B complete.
Python 3.11+, Windows 11

════════════════════════════════════════════════
READ FIRST
════════════════════════════════════════════════
Read core/hardware/serial_controller.py fully before writing anything.
It already has SerialController with simulation mode. Extend it — do not rewrite.

════════════════════════════════════════════════
INSTALL BEFORE THIS SESSION
════════════════════════════════════════════════
pip install pyserial pyautogui pillow pygetwindow pytesseract
pip install pytest pytest-asyncio pytest-cov bandit
# Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki

════════════════════════════════════════════════
PART 1 — HARDWARE / SERIAL
════════════════════════════════════════════════

FILE: core/hardware/serial_controller.py  (MODIFY — extend existing)

Add to existing SerialController:
  async def async_send_command(self, cmd: str, value: str = "") -> dict:
      — run_in_executor wrapping existing sync send_command()
      — return {"success": bool, "response": str, "simulated": bool}

  async def firmware_ping(self) -> bool:
      — send PING command, expect PONG response within 2s
      — in simulation mode: always return True

  async def sensor_read_loop(self, callback, interval: float = 1.0) -> None:
      — loop: await asyncio.sleep(interval); data = await self.async_read_sensor()
      — call callback(data) each iteration
      — stop when self._running is False
      — in simulation: return fake values (temp: 20-25°C, humidity: 40-60%)

FILE: core/hardware/device_registry.py  (NEW)

class DeviceRegistry:
  __init__(): load config/devices.json if exists, else empty dict

  register_device(name, com_port, baud_rate=115200, device_type="arduino") -> None:
      — add to registry, save to config/devices.json

  get_device(name) -> SerialController:
      — return cached instance or create new one
      — if com_port = "SIM": force simulation mode

  list_devices() -> list[dict]:
      — [{name, port, device_type, connected, simulation_mode}]

  Persist to config/devices.json — load on init, save on register

FILE: core/tools/hardware_tools.py  (NEW)

Functions (all async, all return ToolResult from integrations.base):
  send_hardware_command(device_name, command, value="") -> ToolResult
  read_sensor(device_name, sensor_type) -> ToolResult
  list_hardware_devices() -> ToolResult
  ping_device(device_name) -> ToolResult

All use DeviceRegistry — never hardcode port names
Risk levels: read_sensor/list/ping → LOW. send_command → CONFIRM.

Wire into core/tools/builtin_tools.py register_all_tools()
Wire into core/autonomy/risk_evaluator.py (correct sets)
Wire into core/llm/task_planner.py SYSTEM_TOOL_SCHEMA

════════════════════════════════════════════════
PART 2 — GUI AUTOMATION + SCREEN
════════════════════════════════════════════════

FILE: core/tools/screen.py  (NEW)

All functions return ToolResult:
  capture_screen() — PIL screenshot → save to outputs/screenshots/ts.png → return path + dims
  capture_region(x, y, width, height) — region screenshot
  find_text_on_screen(text) — pytesseract OCR → return [{text, x, y, w, h}]
  describe_screen(llm_client=None) — capture → if llava available: send to LLaVA for description
                                   — else: return OCR text dump

All wrapped in try/except ImportError for pyautogui, pytesseract, PIL

FILE: core/tools/gui_control.py  (NEW)

All functions return ToolResult:
  click(x, y, button="left") — save before-screenshot, wait 300ms, click, save after
  double_click(x, y)
  right_click(x, y)
  type_text(text, interval=0.05) — REFUSE if "password" in text.lower()
  hotkey(*keys)
  scroll(x, y, clicks=3, direction="down")
  get_active_window() → {title, x, y, width, height}
  focus_window(title_substring)

SAFETY RULES (hard-coded, not configurable):
  - 300ms delay before every click
  - Validate x,y within screen bounds before any click
  - Save before/after screenshots to outputs/gui_audit/
  - Emergency stop: check for pyautogui.FAILSAFE after every action
  - Never type text containing "password", "passwd", "secret", "token"

Risk levels: capture_screen/get_active_window → LOW. All others → CONFIRM.

Wire into register_all_tools(), risk_evaluator.py, task_planner.py

════════════════════════════════════════════════
PART 3 — SECURITY HARDENING
════════════════════════════════════════════════

FILE: core/tools/builtin_tools.py  (MODIFY)

In _assert_safe_path():
  — reject paths containing ".." after resolving
  — reject symlinks that resolve outside sandbox
  — add max file size check in read_file(): reject > 10MB

FILE: core/execution/dispatcher.py  (MODIFY)

Add input sanitization before any tool call:
  def _sanitize_args(args: dict) -> dict:
      for key, val in args.items():
          if isinstance(val, str):
              val = val.replace("\x00", "")  # strip null bytes
              if len(val) > 4096:
                  val = val[:4096]  # truncate
          args[key] = val
      return args

Add rate limiting (per session):
  _call_count: int = 0
  _call_window_start: float = time.time()
  MAX_CALLS_PER_MINUTE = 30

  Before execute():
      now = time.time()
      if now - self._call_window_start > 60:
          self._call_count = 0
          self._call_window_start = now
      self._call_count += 1
      if self._call_count > MAX_CALLS_PER_MINUTE:
          return error ToolResult("Rate limit exceeded: 30 tool calls/minute")

FILE: core/logger.py  (MODIFY)

Add log rotation in AuditLog:
  Before writing: check if audit.jsonl > 50MB
  If so: rename to audit_TIMESTAMP.jsonl.gz (gzip it), start fresh
  Use gzip stdlib — no external deps

Add scrub_secrets(text: str) -> str helper:
  Replace common secret patterns with [REDACTED]:
  - anything matching env var names from settings.env
  - email passwords, tokens, SIDs in log strings

════════════════════════════════════════════════
PART 4 — TEST SUITE
════════════════════════════════════════════════

Create tests/ files — all use pytest, all async tests use @pytest.mark.asyncio
All external calls (Ollama, APIs, hardware) MUST be mocked — tests pass offline.

FILE: tests/conftest.py  (REPLACE existing)
  Fixtures: tmp_dir, mock_config (ConfigParser), mock_llm (MagicMock), 
            mock_memory (MagicMock), mock_controller

FILE: tests/test_risk_evaluator.py
  — test LOW tools are not blocked
  — test CONFIRM tools require confirmation
  — test CRITICAL tools are hard-blocked
  — test unknown tools default to HIGH
  — test evaluate_plan() with mixed actions
  — test config override of risk levels
  — test RiskLevel ordering (LOW < CONFIRM < HIGH < CRITICAL)
  — test empty action list returns LOW

FILE: tests/test_memory.py
  — test SQLite store_preference
  — test store_conversation
  — test recall_preferences returns correct fields
  — test hybrid_memory.initialize() with and without ChromaDB
  — test index_codebase() hashing (mock filesystem)
  — test store_code_file() AST parsing
  — test store_code_file() with invalid Python (SyntaxError fallback)
  — test mode switches correctly based on semantic availability

FILE: tests/test_agent_loop.py
  — test full loop with mock planner + mock tools → ExecutionTrace
  — test interrupt stops loop
  — test iteration limit triggers stop
  — test risk CRITICAL returns risk_threshold_exceeded
  — test think_blocks saved from LLM response
  — test observation truncation at 800 chars
  — test successful loop sets trace.success = True
  — test failed tool sets trace.success = False

FILE: tests/test_integrations.py
  — test IntegrationLoader scans clients/ dir
  — test loader skips unavailable integrations
  — test loader catches import errors per-plugin
  — test EmailIntegration.is_available() returns False when env vars missing
  — test WhatsAppIntegration.is_available() returns False when twilio missing
  — test CalendarIntegration.is_available() always returns True
  — test ToolResult.to_llm_string() on success and failure
  — test registry.get_tools() returns merged list

FILE: tests/test_profile.py
  — test UserProfileEngine loads defaults when no file exists
  — test save() is atomic (uses tmp file + rename)
  — test update_from_conversation() increments interaction_count
  — test get_communication_style() returns correct string per style
  — test get_system_prompt_injection() stays under 150 tokens
  — test ProfileSynthesizer.should_run() fires at multiples of 20
  — test synthesize() handles LLM returning invalid JSON
  — test synthesize() only updates fields with confidence > 0.6

FILE: tests/test_security.py
  — test path traversal "../../etc/passwd" is blocked
  — test symlink outside sandbox is blocked
  — test file > 10MB is rejected
  — test null bytes stripped from dispatcher args
  — test rate limit triggers after 30 calls/minute
  — test type_text refuses when "password" in text
  — test GUI click outside screen bounds is rejected
  — test dashboard POST without token returns 401
  — test audit log scrub_secrets removes tokens

════════════════════════════════════════════════
DO
════════════════════════════════════════════════
- Extend serial_controller.py — read it first and add only what's missing
- All GUI operations save before/after screenshots for audit trail
- All tests mock external dependencies — must pass with no hardware/internet
- Run bandit -r core/ after security changes — target: no HIGH findings
- Atomic log rotation: gzip old log before starting new one

════════════════════════════════════════════════
DO NOT
════════════════════════════════════════════════
- Do NOT rewrite SerialController — extend it
- Do NOT use shell=True in any subprocess call
- Do NOT allow type_text to type anything containing "password"
- Do NOT expose filesystem browsing in any tool
- Do NOT use unittest — pytest only
- Do NOT skip testing error/failure paths
- Do NOT allow hardware commands at autonomy_level < 2

════════════════════════════════════════════════
FILE DELIVERY ORDER
════════════════════════════════════════════════
 1. core/hardware/serial_controller.py     (modify — extend existing)
 2. core/hardware/device_registry.py       (new)
 3. core/tools/hardware_tools.py           (new)
 4. core/tools/screen.py                   (new)
 5. core/tools/gui_control.py              (new)
 6. core/tools/builtin_tools.py            (modify — path safety + register new tools)
 7. core/execution/dispatcher.py           (modify — sanitization + rate limit)
 8. core/logger.py                         (modify — rotation + secret scrubbing)
 9. core/autonomy/risk_evaluator.py        (modify — add new tool names to correct sets)
10. core/llm/task_planner.py               (modify — add hardware + screen tool schemas)
11. tests/conftest.py                      (replace)
12. tests/test_risk_evaluator.py           (new)
13. tests/test_memory.py                   (new)
14. tests/test_agent_loop.py               (new)
15. tests/test_integrations.py             (new)
16. tests/test_profile.py                  (new)
17. tests/test_security.py                 (new)

════════════════════════════════════════════════
VERIFICATION
════════════════════════════════════════════════
python -c "from core.hardware.device_registry import DeviceRegistry; print('OK')"
python -c "from core.tools.screen import capture_screen; print('OK')"
python -c "from core.tools.gui_control import click; print('OK')"
pip install pytest pytest-asyncio
pytest tests/ -v --tb=short -x
bandit -r core/ -ll
python main.py --help
```

---
---

## COMPLETE CAPABILITY SUMMARY

| Session | Installs | What You Get |
|---------|----------|-------------|
| **A** | httpx aiohttp pvporcupine faster-whisper pyttsx3 sounddevice apscheduler | Voice (speak+listen), Goals+Reminders, Email+WhatsApp+Calendar plugins, User Profile learns you, Health check on startup |
| **B** | fastapi uvicorn websockets jinja2 plyer | Web dashboard at localhost:7070, Multi-model routing, CPU/RAM alerts, Windows notifications |
| **C** | pyserial pyautogui pillow pytesseract pytest bandit | Arduino/hardware control, Screen capture+OCR+GUI automation, Security hardening, Full test suite |

**Total sessions: 3. Do them in order A → B → C.**