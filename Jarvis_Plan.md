# JARVIS — FINAL MASTER ROADMAP V2
# LLM-Optimized | Copy-Paste Ready | One Job Per Session
# Total: 9 Sessions (0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8)
# Run in strict order. Each session has ONE clear job.
# Max ~10 files per session — sized for a single LLM context window.

---

## WHAT ALREADY EXISTS — DO NOT REBUILD
```
core/voice/stt.py              ✅ SpeechToText — Whisper + PyAudio
core/voice/tts.py              ✅ TextToSpeech — Piper TTS
core/voice/wake_word.py        ✅ WakeWordDetector — Porcupine
core/voice/voice_layer.py      ✅ VoiceLayer V2 — complete (do NOT touch)
core/voice/voice_loop.py       ✅ async loop — complete (do NOT touch)
core/autonomy/goal_manager.py  ✅ GoalManager — data classes exist
core/agentic/scheduler.py      ✅ Scheduler — pull-based, exists
core/agentic/belief_state.py   ✅ BeliefState — exists
core/agentic/mission.py        ✅ Mission + MissionStep — exists
core/metrics/confidence.py     ✅ ConfidenceModel — exists
core/introspection/health.py   ✅ HealthReport — exists
core/profile.py                ✅ UserProfileEngine — partial stub
core/synthesis.py              ✅ ProfileSynthesizer — partial stub
integrations/base.py           ✅ BaseIntegration ABC + ToolResult
integrations/registry.py       ✅ api_registry — exists
core/agent/controller.py       ✅ MainController — exists
core/agent/agent_loop.py       ✅ AgentLoopEngine — exists
core/controller_v2.py          ✅ JarvisControllerV2 — exists
core/hardware/serial_controller.py  ✅ SerialController with sim mode — exists
main.py                        ✅ entry point — exists
```

## WHAT IS MISSING (what these sessions build)
```
SESSION 0 → GUI dashboard module (FastAPI + dark web UI)
SESSION 1 → Wire voice into controller
SESSION 2 → Wire goals + scheduler + confidence into agent loop
SESSION 3 → User profile + synthesis (learning who you are)
SESSION 4 → Integration framework (email, calendar, WhatsApp plugins)
SESSION 5 → Startup health check + wire everything into main.py
SESSION 6 → Multi-model routing + proactive notifications
SESSION 7 → Hardware control + screen/GUI automation tools
SESSION 8 → Security hardening + full pytest test suite
```

---
---
---

# SESSION 0 — GUI DASHBOARD
# ONE JOB: Build the dashboard/ module. Nothing else.
# Result: http://localhost:7070 shows a live Iron Man HUD for Jarvis.
# Prerequisite: None. This is the first session.

```
You are an expert Python + web UI engineer.
Project: Jarvis — local agentic AI at D:\AI\Jarvis\
Python 3.11+, Windows 11.
This is SESSION 0. Your ONLY job is the dashboard/ module.
Do NOT touch any core/ files. Do NOT wire voice, goals, or memory yet.
Deliver COMPLETE files only — no partial snippets, no "add this here".

════════════════════════════════════════════════
INSTALL BEFORE STARTING
════════════════════════════════════════════════
pip install fastapi uvicorn websockets jinja2 python-multipart

════════════════════════════════════════════════
CONTEXT — HOW THE GUI WORKS
════════════════════════════════════════════════
The dashboard/ is a self-contained module.
Other sessions will call dashboard.server.update_state() to push live data.
The GUI never imports from core/ — data flows ONE WAY: core → dashboard only.
The GUI is MODULAR: developer changes CSS variables to retheme, adds new
Jinja2 templates for new pages, no server.py changes needed.

════════════════════════════════════════════════
FILES TO CREATE — SESSION 0
════════════════════════════════════════════════

  dashboard/__init__.py
    — empty file, marks as Python package

  dashboard/server.py
    Framework: FastAPI + Jinja2. NO React. NO npm. NO build tools.
    Bind: 127.0.0.1 ONLY — never 0.0.0.0 (hard security rule)
    Port: 7070

    MODULE-LEVEL STATE (how other sessions push live data into GUI):
      import time
      from dataclasses import dataclass, field

      @dataclass
      class JarvisState:
          session_id: str = ""
          state: str = "OFFLINE"
          last_input: str = ""
          last_response: str = ""
          model: str = "unknown"
          memory_count: int = 0
          active_goals: int = 0
          ollama_online: bool = False
          _start_time: float = field(default_factory=time.time)

      _state = JarvisState()
      _controller = None

      def set_controller(controller):
          global _controller
          _controller = controller

      def update_state(**kwargs):
          """Called by any core/ module to push live state to GUI."""
          for k, v in kwargs.items():
              if hasattr(_state, k):
                  setattr(_state, k, v)

    ROUTES:
      GET /
        — render templates/index.html
        — pass state=_state as template context

      GET /memory
        — render templates/memory.html
        — accept ?q= query param
        — open memory/memory.db with sqlite3 (stdlib only, no ORM)
        — SELECT timestamp, category, content FROM memories
          WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 100
        — if DB file missing: pass empty list to template with message

      GET /goals
        — render templates/goals.html
        — if _controller and hasattr(_controller, "goal_manager"):
            load active goals list
          else:
            pass empty list + msg "Goals available after Session 2"

      GET /health
        — return JSON (no template):
          {
            "state": _state.state,
            "ollama_online": _state.ollama_online,
            "memory_count": _state.memory_count,
            "active_goals": _state.active_goals,
            "uptime_seconds": round(time.time() - _state._start_time, 1),
            "model": _state.model
          }

      POST /command
        — accept JSON body: {"text": "..."}
        — require header X-Dashboard-Token
          token = os.environ.get("JARVIS_DASHBOARD_TOKEN", "jarvis")
          if header missing or wrong: return 401 JSON error
        — if _controller is None:
            return {"response": "Jarvis core not connected yet."}
        — call _controller.process(text)
        — update _state.last_input and _state.last_response
        — return {"response": response_text}

      POST /goals/add
        — accept JSON: {"description": "...", "priority": 1}
        — require X-Dashboard-Token header
        — if _controller and hasattr(_controller, "goal_manager"):
            call goal_manager.create(description=..., priority=...)
            return {"ok": True}
          else:
            return {"error": "Goal manager not wired yet"}

      POST /goals/complete/{goal_id}
        — require X-Dashboard-Token
        — call goal_manager.complete(goal_id) if available
        — return {"ok": True}

      WS /ws
        — on connect: start sending JSON every 2 seconds:
          {
            "state": _state.state,
            "last_response": _state.last_response,
            "last_input": _state.last_input,
            "session_id": _state.session_id,
            "memory_count": _state.memory_count,
            "active_goals": _state.active_goals,
            "ollama_online": _state.ollama_online,
            "uptime_seconds": round(time.time() - _state._start_time, 1),
            "model": _state.model
          }
        — on disconnect: silently close, never crash server
        — use asyncio.sleep(2) between pushes

  dashboard/templates/base.html
    Jinja2 base layout. All other templates extend this.
    Must contain:
      - <head> with CSS link to /static/style.css and meta charset
      - <nav> with links: Home (/) | Memory (/memory) | Goals (/goals) | Health (/health)
      - {% block content %}{% endblock %}
      - <script src="/static/app.js"></script> at bottom of body

  dashboard/templates/index.html  (extends base.html)
    Main dashboard. Vanilla JS connects to WS /ws and updates DOM live.

    MUST CONTAIN THESE UI SECTIONS:
      1. STATE BADGE — large, dominant visual element
           OFFLINE  → gray background
           IDLE     → green background
           THINKING → yellow, animated CSS pulse
           ACTING   → orange, animated CSS pulse
           SPEAKING → blue, animated CSS pulse
           ERROR    → red background

      2. LAST RESPONSE PANEL
           Scrollable div. Monospace font. Shows _state.last_response.
           Auto-scrolls to bottom when new content arrives via WebSocket.

      3. LAST INPUT LINE
           Small line above response panel showing what the user said.

      4. COMMAND INPUT
           Text input + "Send" button.
           On submit: POST to /command with X-Dashboard-Token header.
           Show loading spinner while waiting.
           Enter key triggers submit.
           Clear input after response received.

      5. SESSION INFO BAR (bottom or top strip)
           Shows: Session ID | Model | Uptime | Memory entries | Active goals | Ollama ●/●

      6. DESIGN:
           Dark theme. Background #0d0d1a. Accent #4ecca3. Text #e0e0e0.
           Monospace font throughout.
           Feels like a terminal / Iron Man HUD, not a web app.
           State badge is the biggest visual element on the page.
           No external CDN links. No external JS libraries.

  dashboard/templates/memory.html  (extends base.html)
    Shows memory entries from memory.db.
    Search box at top — submits as GET /memory?q=searchterm.
    Results table: Timestamp | Category | Content (truncated to 180 chars).
    If no DB or no results: show placeholder message.
    Link back to / in nav (inherited from base.html).

  dashboard/templates/goals.html  (extends base.html)
    Shows active goals.
    If goal_manager not wired yet: show placeholder.
    Table: Priority | Description | Created | Status.
    "Add Goal" form at bottom: description text + priority 1-4 dropdown.
    "Complete" button per row.

  dashboard/static/style.css
    Use CSS custom properties at the top (easy retheme — change 3 lines):
      :root {
        --bg:      #0d0d1a;
        --bg2:     #12122a;
        --accent:  #4ecca3;
        --text:    #e0e0e0;
        --danger:  #ff6b6b;
        --warn:    #ffd93d;
        --font:    'Courier New', Courier, monospace;
      }
    Style: body, nav, .state-badge, .response-panel, .cmd-input,
           table, th, td, .btn, .btn-sm, .spinner, .info-bar
    Animations:
      @keyframes pulse — scale 1.0 → 1.05 → 1.0, loop
      .state-THINKING, .state-ACTING, .state-SPEAKING use pulse animation

  dashboard/static/app.js
    Keep under 100 lines. No jQuery. No frameworks. Pure DOM.

    On DOMContentLoaded:
      1. Connect: ws = new WebSocket("ws://localhost:7070/ws")
      2. ws.onmessage: parse JSON, update:
           document.getElementById("state-badge").textContent = data.state
           document.getElementById("state-badge").className = "state-badge state-" + data.state
           document.getElementById("last-response").textContent = data.last_response
           document.getElementById("session-info").textContent = build info string
           (and other fields)
           auto-scroll response panel to bottom
      3. ws.onclose / ws.onerror:
           update state badge to OFFLINE
           setTimeout(reconnect, 3000)  // retry every 3s
      4. Command form:
           prevent default submit
           show spinner
           fetch("http://localhost:7070/command", {
             method: "POST",
             headers: {
               "Content-Type": "application/json",
               "X-Dashboard-Token": "jarvis"
             },
             body: JSON.stringify({text: inputValue})
           })
           on response: hide spinner, clear input
           (WebSocket will push the updated last_response automatically)

════════════════════════════════════════════════
HOW OTHER SESSIONS USE THIS MODULE
════════════════════════════════════════════════
After Session 0, any core/ file can call:
  try:
      from dashboard.server import update_state
      update_state(state="THINKING")
      update_state(last_response="Done.", memory_count=42)
  except ImportError:
      pass   # dashboard not running — silently skip

This is the ONLY communication pattern. Never import core/ from dashboard/.

════════════════════════════════════════════════
DO — SESSION 0
════════════════════════════════════════════════
- Bind 127.0.0.1 ONLY — this is a hard security rule, never 0.0.0.0
- All POST routes validate X-Dashboard-Token before doing anything
- WebSocket reconnects automatically on disconnect (handled in app.js)
- Memory browser degrades gracefully if memory.db doesn't exist
- Goals page degrades gracefully if goal_manager not wired yet
- CSS uses variables — the whole theme changes by editing 3 lines
- Templates use Jinja2 block inheritance from base.html

════════════════════════════════════════════════
DO NOT — SESSION 0
════════════════════════════════════════════════
- Do NOT touch any core/ files in this session
- Do NOT use React, Vue, npm, webpack, or any build tool
- Do NOT use external CDN JS libraries (no jQuery, no axios)
- Do NOT use SQLAlchemy — sqlite3 stdlib only
- Do NOT bind to 0.0.0.0 under any circumstance
- Do NOT expose filesystem paths in any API response

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 0
════════════════════════════════════════════════
1.  dashboard/__init__.py
2.  dashboard/static/style.css
3.  dashboard/templates/base.html
4.  dashboard/templates/index.html
5.  dashboard/templates/memory.html
6.  dashboard/templates/goals.html
7.  dashboard/static/app.js
8.  dashboard/server.py

════════════════════════════════════════════════
VERIFICATION — SESSION 0
════════════════════════════════════════════════
python -c "from dashboard.server import app, update_state; print('OK')"
python -c "import uvicorn; from dashboard.server import app; print('OK')"

# Start dashboard standalone to verify UI:
python -c "
import uvicorn
from dashboard.server import app
uvicorn.run(app, host='127.0.0.1', port=7070)
"
# Open http://localhost:7070 in browser
# Should see: dark UI, OFFLINE badge (gray), nav links, command input box
# /health should return JSON
curl http://localhost:7070/health
```

---
---

# SESSION 1 — WIRE VOICE INTO CONTROLLER
# ONE JOB: Connect VoiceLayer to JarvisControllerV2. Wire state updates to GUI.
# Result: python main.py --voice → Jarvis speaks and listens.
# Prerequisite: Session 0 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Session 0 (GUI dashboard) is complete. dashboard/ module exists and works.
Python 3.11+, Windows 11, Ollama at http://localhost:11434.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
Read these files completely. Use ACTUAL method names found in them.
Do not invent method names. Do not rewrite these files.

  core/voice/voice_layer.py     — VoiceLayer class, find: __init__, start(), stop()
  core/voice/voice_loop.py      — find the async loop orchestrator method names
  core/voice/stt.py             — SpeechToText, find transcribe() or equivalent
  core/voice/tts.py             — TextToSpeech, find speak() or equivalent
  core/controller_v2.py         — JarvisControllerV2, find __init__, process(), run_cli()
  main.py                       — find how controller is instantiated and started

════════════════════════════════════════════════
INSTALL BEFORE STARTING
════════════════════════════════════════════════
pip install pvporcupine pvrecorder faster-whisper pyttsx3 sounddevice soundfile numpy

════════════════════════════════════════════════
SESSION 1 GOAL — VOICE WIRING
════════════════════════════════════════════════

FILE: core/controller_v2.py  (MODIFY — extend, keep all existing code)

In JarvisControllerV2.__init__(self, config=None, voice=False, **kwargs):
  Add these lines (after existing __init__ body):
    self.voice_enabled = voice
    self._voice_layer = None
    if voice:
        try:
            from core.voice.voice_layer import VoiceLayer
            self._voice_layer = VoiceLayer(
                config=config,
                text_handler=self._voice_text_handler
            )
            logger.info("VoiceLayer initialized")
        except ImportError as e:
            logger.warning(f"Voice unavailable: {e}")
        except Exception as e:
            logger.warning(f"VoiceLayer init failed: {e}")

Add method to JarvisControllerV2:
  async def _voice_text_handler(self, text: str) -> str:
      """Called by VoiceLayer when speech is recognized."""
      try:
          from dashboard.server import update_state
          update_state(state="THINKING", last_input=text)
      except ImportError:
          pass
      response = self.process(text)
      try:
          from dashboard.server import update_state
          update_state(state="IDLE", last_response=response)
      except ImportError:
          pass
      return response

Modify existing run_cli(self) method (or run() — use actual method name):
  At the START of the method, add:
    if self._voice_layer is not None:
        logger.info("Starting in voice mode...")
        try:
            from dashboard.server import update_state
            update_state(state="IDLE")
        except ImportError:
            pass
        await self._voice_layer.start()   # use actual start() method from voice_layer.py
        return   # voice loop handles everything, CLI loop not needed in voice mode
  Keep existing CLI loop below exactly as is (runs when voice is False)

Add to shutdown(self) method (or create it if missing):
  if self._voice_layer is not None:
      try:
          await self._voice_layer.stop()   # use actual stop() method
      except Exception as e:
          logger.warning(f"VoiceLayer stop error: {e}")

Also: in the existing process() method (find it — it's the method that takes user_input):
  At the START of process(), add:
    try:
        from dashboard.server import update_state
        update_state(state="THINKING", last_input=user_input)
    except ImportError:
        pass

  At the END of process() just before returning response, add:
    try:
        from dashboard.server import update_state
        update_state(state="IDLE", last_response=response)
    except ImportError:
        pass

FILE: main.py  (MODIFY — add --voice flag)

Find the argparse section. Add:
  parser.add_argument("--voice", action="store_true",
                      help="Start Jarvis in voice mode (speak + listen)")

Find where JarvisControllerV2 (or the main controller) is instantiated.
Pass voice flag:
  controller = JarvisControllerV2(config=config, voice=args.voice)

Also add --gui flag wiring here:
  parser.add_argument("--gui", action="store_true",
                      help="Start web dashboard at http://localhost:7070")
  if args.gui:
      import threading
      import uvicorn
      from dashboard.server import app as dashboard_app, set_controller, update_state
      set_controller(controller)
      try:
          update_state(
              session_id=getattr(controller, "session_id", "session-1"),
              model=getattr(getattr(controller, "llm", None), "model", "unknown"),
              state="IDLE"
          )
      except Exception:
          pass
      def _run_dashboard():
          uvicorn.run(dashboard_app, host="127.0.0.1", port=7070, log_level="warning")
      t = threading.Thread(target=_run_dashboard, daemon=True)
      t.start()
      print("Dashboard running at: http://localhost:7070")

════════════════════════════════════════════════
DO — SESSION 1
════════════════════════════════════════════════
- Read voice_layer.py and controller_v2.py fully before writing anything
- Use ACTUAL method names found in those files
- Wrap all dashboard imports in try/except ImportError — GUI may not be running
- If VoiceLayer import fails: log warning and continue in CLI mode (never crash)
- Keep all existing JarvisControllerV2 methods intact — add only, never remove

════════════════════════════════════════════════
DO NOT — SESSION 1
════════════════════════════════════════════════
- Do NOT rewrite voice_layer.py — it is complete
- Do NOT rewrite voice_loop.py — it is complete
- Do NOT rewrite stt.py, tts.py, or wake_word.py
- Do NOT remove any existing code from controller_v2.py
- Do NOT use asyncio.get_event_loop() — use get_running_loop() in async contexts

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 1
════════════════════════════════════════════════
1. core/controller_v2.py   (modify — add voice wiring + dashboard state push)
2. main.py                 (modify — add --voice and --gui flags)

════════════════════════════════════════════════
VERIFICATION — SESSION 1
════════════════════════════════════════════════
python main.py --help
# Must show: --voice, --gui flags

python main.py --gui &
# Open http://localhost:7070
# State badge should show IDLE (green) not OFFLINE

python main.py --gui
# Type something in CLI → state badge flickers THINKING → IDLE
# Last response panel updates
```

---
---

# SESSION 2 — WIRE GOALS + SCHEDULER + CONFIDENCE
# ONE JOB: Goals persist. Scheduler fires reminders. Confidence gates tool use.
# Result: "remind me to X" → goal saved. Overdue goals → terminal notification.
# Prerequisite: Sessions 0, 1 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0 and 1 are complete.
Python 3.11+, Windows 11, Ollama at http://localhost:11434.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
  core/autonomy/goal_manager.py  — read fully, note EXACT class name and method names
  core/agentic/scheduler.py      — read fully, note EXACT class name and method names
  core/metrics/confidence.py     — read fully, note EXACT class name and update() method
  core/agent/agent_loop.py       — read fully, find where tools are dispatched
  core/controller_v2.py          — read the current state after Session 1 changes

════════════════════════════════════════════════
INSTALL BEFORE STARTING
════════════════════════════════════════════════
pip install apscheduler

════════════════════════════════════════════════
SESSION 2 GOAL — GOALS + SCHEDULER + CONFIDENCE
════════════════════════════════════════════════

PART A: Wire goals + scheduler into controller_v2.py

FILE: core/controller_v2.py  (MODIFY — extend, keep all existing Session 1 code)

In JarvisControllerV2.__init__, add after existing init body:
  from core.autonomy.goal_manager import <ActualClassName>
  from core.agentic.scheduler import <ActualClassName>
  self.goal_manager = <ActualClassName>()
  self.scheduler = <ActualClassName>()

  # NOTE: Replace <ActualClassName> with the real class names from the files.

In process(self, user_input: str) method, add BEFORE calling LLM:
  # Goal intent detection
  lower = user_input.lower()
  if any(kw in lower for kw in ("remind me", "set goal", "schedule", "don't forget", "remember to")):
      # Parse description: strip intent phrases, keep the rest
      description = user_input
      for kw in ("remind me to", "set goal", "schedule", "don't forget to", "remember to"):
          description = description.replace(kw, "").strip()
      description = description.strip(" .?!")
      if description:
          self.goal_manager.<create_method>(description=description)
          # use actual create method name from goal_manager.py
          response = f"✓ Goal set: {description}"
          try:
              from dashboard.server import update_state
              update_state(active_goals=len(self.goal_manager.<list_method>()))
          except ImportError:
              pass
          return response

  if any(kw in lower for kw in ("what are my goals", "show goals", "list goals", "my goals")):
      goals = self.goal_manager.<list_active_method>()
      # use actual list method name from goal_manager.py
      if not goals:
          return "No active goals."
      lines = [f"• [{g.priority}] {g.description}" for g in goals]
      # adjust field names to match actual Goal dataclass fields
      return "Active goals:\n" + "\n".join(lines)

Add method to JarvisControllerV2:
  async def _check_due_goals(self) -> None:
      """Background task: check for overdue goals every 5 minutes."""
      while True:
          await asyncio.sleep(300)   # 5 minutes
          try:
              due_items = self.scheduler.<due_method>()
              # use actual method from scheduler.py
              for item in due_items:
                  msg = f"⏰ Due: {item}"
                  print(f"\n[JARVIS] {msg}")
                  if self._voice_layer is not None:
                      try:
                          await self._voice_layer.<speak_method>(msg)
                          # use actual speak method from voice_layer.py
                      except Exception:
                          pass
              # Also push active goal count to GUI
              try:
                  from dashboard.server import update_state
                  update_state(active_goals=len(self.goal_manager.<list_method>()))
              except ImportError:
                  pass
          except Exception as e:
              logger.warning(f"Goal check error: {e}")

In run_cli() (or run() — use actual method), add this BEFORE the main loop starts:
  asyncio.create_task(self._check_due_goals())

PART B: Wire confidence into agent_loop.py

FILE: core/agent/agent_loop.py  (MODIFY — extend, keep all existing code)

In AgentLoopEngine.__init__, add:
  from core.metrics.confidence import ConfidenceModel
  self.confidence = ConfidenceModel()

Find the section in run() where IntentClassifier (or equivalent) returns a score.
After that, add:
  # Update confidence with intent clarity
  intent_score = getattr(classification_result, "confidence", 0.5)
  # use actual attribute name from your IntentClassifier result
  self.confidence.update("intent_clarity", intent_score)

Find the section in run() where ToolObservation results come back.
After each observation, add:
  _tool_success = 1.0 if obs.execution_status == "success" else 0.0
  self.confidence.update("tool_reliability", _tool_success)

Find the section where CONFIRM-level tools are about to execute.
Before executing, add:
  if self.confidence.score() < 0.4:
      force_confirmation = True   # always ask user when confidence is low

Also in agent_loop.py:
  Find MAX_ITERATIONS (or similar constant) and change from hardcoded 5 to:
    self.max_iterations = config.getint("agent", "max_iterations", fallback=10)
    # or just: self.max_iterations = 10  if config not available

  Add to ExecutionTrace dataclass (find it and add a field):
    think_blocks: list = field(default_factory=list)

  In the _reflect() method (find it — handles LLM reflection after tool use):
    Add after getting LLM response text:
      import re
      think_matches = re.findall(r"<think>(.*?)</think>", response_text, re.DOTALL)
      trace.think_blocks = think_matches

  Add this helper function BEFORE the class definition:
    def _truncate_obs(text: str, max_chars: int = 800) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        omitted = len(text) - max_chars
        return text[:half] + f"\n...[{omitted} chars omitted]...\n" + text[-half:]

  Find where observation output_summary is passed to the reflection prompt.
  Wrap it: obs_text = _truncate_obs(obs.output_summary)
  Use obs_text instead of obs.output_summary in the prompt.

  Find REFLECT_SYSTEM_PROMPT (or equivalent constant). Replace its value with:
    "You are Jarvis, an expert AI assistant. Review the executed plan and observations.
     If any tool failed: state the root cause first, then the fix.
     If successful: summarize concisely what was accomplished.
     Be direct and technical. No filler phrases. Address the user in second person."

════════════════════════════════════════════════
DO — SESSION 2
════════════════════════════════════════════════
- Read goal_manager.py and scheduler.py — use ACTUAL method names
- Goals persist to memory/goals.json (check if GoalManager already does this — if yes, leave it)
- If goals.json doesn't exist: GoalManager should create it on first save
- _check_due_goals runs as asyncio.create_task — never blocking
- Wrap all dashboard state pushes in try/except ImportError

════════════════════════════════════════════════
DO NOT — SESSION 2
════════════════════════════════════════════════
- Do NOT rewrite GoalManager or Scheduler — use what exists
- Do NOT create new Goal dataclasses — use the existing ones
- Do NOT use asyncio.get_event_loop() — use create_task() inside async methods
- Do NOT block the main loop in _check_due_goals

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 2
════════════════════════════════════════════════
1. core/agent/agent_loop.py    (modify — confidence, think_blocks, truncation, reflect prompt)
2. core/controller_v2.py       (modify — goal wiring, scheduler, _check_due_goals)

════════════════════════════════════════════════
VERIFICATION — SESSION 2
════════════════════════════════════════════════
python -c "from core.autonomy.goal_manager import *; print('OK')"
python -c "from core.agentic.scheduler import *; print('OK')"
python -c "from core.metrics.confidence import ConfidenceModel; c=ConfidenceModel(); print('OK')"
python main.py --gui
# Type: "remind me to call mom tomorrow"
# Should return: "✓ Goal set: call mom tomorrow"
# /goals page in dashboard should show the new goal
```

---
---

# SESSION 3 — USER PROFILE + SYNTHESIS
# ONE JOB: Jarvis learns who you are across sessions. Profile adapts your responses.
# Result: Jarvis becomes more personalised after every 20 conversations.
# Prerequisite: Sessions 0, 1, 2 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0, 1, 2 are complete.
Python 3.11+, Windows 11, Ollama at http://localhost:11434.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
  core/profile.py        — read the partial UserProfileEngine stub
  core/synthesis.py      — read the partial ProfileSynthesizer stub
  core/controller_v2.py  — read the current state after Sessions 1+2 changes
  core/llm/client.py     — find where the system prompt is built (_build_system or similar)

════════════════════════════════════════════════
SESSION 3 GOAL — PROFILE + SYNTHESIS
════════════════════════════════════════════════

FILE: core/profile.py  (REPLACE — keep class name UserProfileEngine exactly)

Write a complete replacement. The new file must have:

  class UserProfileEngine:
      PROFILE_PATH = Path("memory/user_profile.json")

      DEFAULTS = {
          "name": "User",
          "communication_style": "casual",   # casual | formal | technical
          "expertise_level": "intermediate",  # beginner | intermediate | advanced | expert
          "preferred_topics": [],
          "timezone": "UTC",
          "language": "en",
          "interaction_count": 0,
          "first_seen": None,
          "last_seen": None
      }

      def __init__(self):
          self._data = dict(self.DEFAULTS)
          self._load()

      def _load(self) -> None:
          try:
              if self.PROFILE_PATH.exists():
                  with open(self.PROFILE_PATH, "r", encoding="utf-8") as f:
                      loaded = json.load(f)
                  for k, v in loaded.items():
                      if k in self._data:
                          self._data[k] = v
          except Exception as e:
              logging.getLogger(__name__).warning(f"Profile load failed: {e}")
              self._data = dict(self.DEFAULTS)

      def save(self) -> None:
          """Atomic write — never corrupt on crash."""
          self.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
          tmp = self.PROFILE_PATH.with_suffix(".tmp")
          try:
              with open(tmp, "w", encoding="utf-8") as f:
                  json.dump(self._data, f, indent=2, ensure_ascii=False)
              os.replace(tmp, self.PROFILE_PATH)
          except Exception as e:
              logging.getLogger(__name__).error(f"Profile save failed: {e}")

      def update_from_conversation(self, user_text: str, jarvis_response: str) -> None:
          now = datetime.now().isoformat()
          self._data["interaction_count"] = self._data.get("interaction_count", 0) + 1
          self._data["last_seen"] = now
          if self._data.get("first_seen") is None:
              self._data["first_seen"] = now
          # Detect name from single clear statement only
          lower = user_text.lower()
          for pattern in ("my name is ", "i am ", "i'm ", "call me "):
              if pattern in lower:
                  idx = lower.index(pattern) + len(pattern)
                  candidate = user_text[idx:].split()[0].strip(".,!?")
                  if 2 <= len(candidate) <= 30 and candidate.isalpha():
                      self._data["name"] = candidate
                      break
          self.save()

      def apply_delta(self, delta: dict, min_confidence: float = 0.6) -> list:
          """Apply synthesis delta. Returns list of updated field names."""
          updated = []
          for field, val in delta.items():
              if isinstance(val, dict):
                  confidence = val.get("confidence", 0.0)
                  value = val.get("value")
              else:
                  confidence = 1.0
                  value = val
              if confidence >= min_confidence and field in self.DEFAULTS and value is not None:
                  self._data[field] = value
                  updated.append(field)
          if updated:
              self.save()
          return updated

      def get_system_prompt_injection(self) -> str:
          """Max ~150 tokens. Injected into every LLM system prompt."""
          parts = [f"User: {self._data['name']}."]
          parts.append(f"Style: {self._data['communication_style']}.")
          parts.append(f"Level: {self._data['expertise_level']}.")
          if self._data.get("preferred_topics"):
              topics = ", ".join(self._data["preferred_topics"][:3])
              parts.append(f"Interests: {topics}.")
          return " ".join(parts)

      def get_communication_style(self) -> str:
          style = self._data.get("communication_style", "casual")
          return {
              "formal":    "Be precise and professional. Use formal language.",
              "casual":    "Be friendly and conversational. Keep it natural.",
              "technical": "Be detailed and technical. Use correct terminology.",
          }.get(style, "Be helpful and clear.")

      @property
      def interaction_count(self) -> int:
          return self._data.get("interaction_count", 0)

FILE: core/synthesis.py  (REPLACE — keep class name ProfileSynthesizer exactly)

Write a complete replacement:

  SYNTHESIS_SYSTEM = """You are analyzing conversation history to understand a user's
  communication style and expertise level. Respond ONLY with a JSON object.
  No preamble. No markdown. No code fences. Raw JSON only.
  Format:
  {
    "communication_style": {"value": "casual|formal|technical", "confidence": 0.0-1.0},
    "expertise_level": {"value": "beginner|intermediate|advanced|expert", "confidence": 0.0-1.0},
    "preferred_topics": {"value": ["topic1", "topic2"], "confidence": 0.0-1.0},
    "name": {"value": "FirstName", "confidence": 0.0-1.0}
  }
  Only include fields you are confident about. Omit fields you cannot determine.
  """

  class ProfileSynthesizer:
      def __init__(self, llm):
          self._llm = llm

      def should_run(self, profile: UserProfileEngine) -> bool:
          count = profile.interaction_count
          return count > 0 and count % 20 == 0

      async def synthesize(self, recent_conversations: list, profile: UserProfileEngine) -> dict:
          """Analyze last N conversations. Update profile fields where confident."""
          if not recent_conversations:
              return {"updated_fields": [], "delta": {}}

          convo_text = "\n\n".join(recent_conversations[-20:])
          prompt = f"Analyze these conversations and extract user profile signals:\n\n{convo_text}"

          try:
              raw = await asyncio.get_running_loop().run_in_executor(
                  None,
                  lambda: self._llm.complete(prompt, system=SYNTHESIS_SYSTEM, task_type="synthesis")
              )
              # Strip markdown fences if present
              clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
              delta = json.loads(clean)
              updated = profile.apply_delta(delta)
              return {"updated_fields": updated, "delta": delta}
          except json.JSONDecodeError as e:
              logging.getLogger(__name__).warning(f"Synthesis JSON parse failed: {e}")
              return {"updated_fields": [], "delta": {}, "error": "json_parse_failed"}
          except Exception as e:
              logging.getLogger(__name__).warning(f"Synthesis failed: {e}")
              return {"updated_fields": [], "delta": {}, "error": str(e)}

FILE: core/controller_v2.py  (MODIFY — add profile wiring, keep all existing Session 1+2 code)

In JarvisControllerV2.__init__, add:
  from core.profile import UserProfileEngine
  from core.synthesis import ProfileSynthesizer
  self.profile = UserProfileEngine()
  self.synthesizer = ProfileSynthesizer(self.llm)
  self._conversation_buffer: list[str] = []

In process() method, AFTER getting the response and BEFORE returning it, add:
  # Update profile
  self.profile.update_from_conversation(user_input, response)
  self._conversation_buffer.append(f"User: {user_input}\nJarvis: {response}")
  # Run synthesis every 20 conversations (as background task)
  if self.synthesizer.should_run(self.profile):
      asyncio.create_task(
          self.synthesizer.synthesize(self._conversation_buffer[-20:], self.profile)
      )
      self._conversation_buffer.clear()

Find wherever the LLM system prompt is assembled (look for _build_system or system= param).
Inject profile data:
  profile_injection = self.profile.get_system_prompt_injection()
  style_instruction = self.profile.get_communication_style()
  # Add these to the system prompt string before sending to LLM
  # The total injection should stay under 200 tokens

Also wire into core/llm/client.py if that's where system prompts are built:
  Find the method that builds the system prompt.
  Add profile fields if a profile attribute is available on the client.

════════════════════════════════════════════════
DO — SESSION 3
════════════════════════════════════════════════
- Atomic saves only: write to .tmp then os.replace() — never write directly
- Profile is human-readable JSON — user may hand-edit it
- Synthesis runs as asyncio.create_task() — never awaited in main flow
- JSON parse errors in synthesis must be caught and logged, never crash
- Profile injection into LLM must be under 200 tokens total

════════════════════════════════════════════════
DO NOT — SESSION 3
════════════════════════════════════════════════
- Do NOT store passwords, health data, financial data in profile
- Do NOT update style/expertise from a single message — only from synthesis
- Do NOT run synthesizer more than once per 20 interactions
- Do NOT let synthesis block the user's response — background task only
- Do NOT use asyncio.get_event_loop() — use get_running_loop() inside async

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 3
════════════════════════════════════════════════
1. core/profile.py          (replace — keep class name UserProfileEngine)
2. core/synthesis.py        (replace — keep class name ProfileSynthesizer)
3. core/controller_v2.py    (modify — wire profile + synthesis)

════════════════════════════════════════════════
VERIFICATION — SESSION 3
════════════════════════════════════════════════
python -c "
from core.profile import UserProfileEngine
p = UserProfileEngine()
print(p.get_communication_style())
print(p.get_system_prompt_injection())
print('interaction_count:', p.interaction_count)
"
python -c "from core.synthesis import ProfileSynthesizer; print('OK')"
python main.py --gui
# Talk for a few turns, check memory/user_profile.json is created and updated
```

---
---

# SESSION 4 — INTEGRATION FRAMEWORK (EMAIL, CALENDAR, WHATSAPP)
# ONE JOB: Build the plugin loader + 3 integration clients.
# Result: Jarvis can send email, check calendar, send WhatsApp (if configured).
# Prerequisite: Sessions 0–3 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0–3 are complete.
Python 3.11+, Windows 11, Ollama at http://localhost:11434.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
  integrations/base.py      — read BaseIntegration ABC and ToolResult fully
  integrations/registry.py  — read api_registry, find register() and list_schemas() methods
  core/execution/dispatcher.py — find execute() — this is where tool routing happens
  core/llm/task_planner.py     — find SYSTEM_TOOL_SCHEMA and where it's sent to LLM

════════════════════════════════════════════════
SESSION 4 GOAL — INTEGRATION FRAMEWORK
════════════════════════════════════════════════

FILE: integrations/base.py  (MODIFY — keep ALL existing code, add to ABC only)

Add these to the BaseIntegration ABC (do not remove existing fields):
  name: str = ""
  description: str = ""
  required_config: list = []   # list of env var names that must be set

  @abstractmethod
  def is_available(self) -> bool:
      """Return True if deps installed AND required env vars are set.
         NEVER raise — return False silently if anything is missing."""
      ...

  @abstractmethod
  def get_tools(self) -> list:
      """Return list of tool schema dicts in SYSTEM_TOOL_SCHEMA format."""
      ...

FILE: integrations/loader.py  (NEW)

  import importlib, inspect, logging
  from pathlib import Path
  from integrations.base import BaseIntegration

  logger = logging.getLogger(__name__)

  class IntegrationLoader:
      def load_all(self, config, registry) -> dict:
          loaded = []
          skipped = []
          clients_dir = Path(__file__).parent / "clients"
          if not clients_dir.exists():
              return {"loaded": [], "skipped": ["clients/ dir not found"]}

          for py_file in clients_dir.glob("*.py"):
              if py_file.name.startswith("_"):
                  continue
              module_name = f"integrations.clients.{py_file.stem}"
              try:
                  module = importlib.import_module(module_name)
                  for _, cls in inspect.getmembers(module, inspect.isclass):
                      if issubclass(cls, BaseIntegration) and cls is not BaseIntegration:
                          try:
                              instance = cls()
                              if instance.is_available():
                                  registry.register(instance)
                                  loaded.append(instance.name or cls.__name__)
                                  logger.info(f"Integration loaded: {instance.name}")
                              else:
                                  skipped.append(f"{cls.__name__} (not available)")
                                  logger.debug(f"Integration skipped: {cls.__name__}")
                          except Exception as e:
                              skipped.append(f"{cls.__name__} (init error: {e})")
                              logger.warning(f"Integration init failed {cls.__name__}: {e}")
              except Exception as e:
                  skipped.append(f"{py_file.stem} (import error: {e})")
                  logger.warning(f"Integration import failed {py_file.stem}: {e}")

          return {"loaded": loaded, "skipped": skipped}

FILE: integrations/clients/__init__.py  (NEW — empty)

FILE: integrations/clients/email.py  (NEW)

  import os, smtplib, imaplib, email as email_lib
  from integrations.base import BaseIntegration, ToolResult

  class EmailIntegration(BaseIntegration):
      name = "email"
      description = "Send and read emails via SMTP/IMAP"
      required_config = ["EMAIL_ADDRESS", "EMAIL_PASSWORD", "SMTP_HOST", "IMAP_HOST"]

      def is_available(self) -> bool:
          try:
              import smtplib, imaplib  # stdlib — always available
              return all(os.environ.get(k) for k in self.required_config)
          except Exception:
              return False

      def get_tools(self) -> list:
          return [
              {
                  "name": "send_email",
                  "description": "Send an email",
                  "parameters": {
                      "to": {"type": "string", "description": "Recipient email"},
                      "subject": {"type": "string"},
                      "body": {"type": "string"}
                  }
              },
              {
                  "name": "read_emails",
                  "description": "Read recent emails from inbox",
                  "parameters": {
                      "folder": {"type": "string", "default": "INBOX"},
                      "limit": {"type": "integer", "default": 10}
                  }
              },
              {
                  "name": "search_emails",
                  "description": "Search emails by keyword",
                  "parameters": {
                      "query": {"type": "string"}
                  }
              }
          ]

      def execute(self, tool_name: str, **kwargs) -> ToolResult:
          try:
              if tool_name == "send_email":
                  return self._send_email(**kwargs)
              elif tool_name == "read_emails":
                  return self._read_emails(**kwargs)
              elif tool_name == "search_emails":
                  return self._search_emails(**kwargs)
              return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
          except Exception as e:
              return ToolResult(success=False, error=str(e))

      def _send_email(self, to, subject, body) -> ToolResult:
          addr = os.environ["EMAIL_ADDRESS"]
          pwd = os.environ["EMAIL_PASSWORD"]
          host = os.environ["SMTP_HOST"]
          port = int(os.environ.get("SMTP_PORT", 587))
          from email.mime.text import MIMEText
          msg = MIMEText(body)
          msg["Subject"] = subject
          msg["From"] = addr
          msg["To"] = to
          with smtplib.SMTP(host, port, timeout=10) as s:
              s.starttls()
              s.login(addr, pwd)
              s.send_message(msg)
          return ToolResult(success=True, data={"sent_to": to})

      def _read_emails(self, folder="INBOX", limit=10) -> ToolResult:
          addr = os.environ["EMAIL_ADDRESS"]
          pwd = os.environ["EMAIL_PASSWORD"]
          host = os.environ["IMAP_HOST"]
          with imaplib.IMAP4_SSL(host, timeout=10) as m:
              m.login(addr, pwd)
              m.select(folder)
              _, data = m.search(None, "ALL")
              ids = data[0].split()[-limit:]
              results = []
              for eid in reversed(ids):
                  _, edata = m.fetch(eid, "(RFC822)")
                  msg = email_lib.message_from_bytes(edata[0][1])
                  results.append({
                      "from": msg.get("From"),
                      "subject": msg.get("Subject"),
                      "date": msg.get("Date")
                  })
          return ToolResult(success=True, data={"emails": results})

      def _search_emails(self, query) -> ToolResult:
          addr = os.environ["EMAIL_ADDRESS"]
          pwd = os.environ["EMAIL_PASSWORD"]
          host = os.environ["IMAP_HOST"]
          with imaplib.IMAP4_SSL(host, timeout=10) as m:
              m.login(addr, pwd)
              m.select("INBOX")
              _, data = m.search(None, f'SUBJECT "{query}"')
              ids = data[0].split()
              return ToolResult(success=True, data={"matches": len(ids), "ids": ids[-10:]})

FILE: integrations/clients/whatsapp.py  (NEW)

  import os
  from integrations.base import BaseIntegration, ToolResult

  class WhatsAppIntegration(BaseIntegration):
      name = "whatsapp"
      description = "Send WhatsApp messages via Twilio"
      required_config = ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"]

      def is_available(self) -> bool:
          try:
              import twilio
              return all(os.environ.get(k) for k in self.required_config)
          except ImportError:
              return False

      def get_tools(self) -> list:
          return [{
              "name": "send_whatsapp",
              "description": "Send a WhatsApp message",
              "parameters": {
                  "to": {"type": "string", "description": "Recipient phone number with country code"},
                  "message": {"type": "string"}
              }
          }]

      def execute(self, tool_name: str, **kwargs) -> ToolResult:
          if tool_name != "send_whatsapp":
              return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
          try:
              from twilio.rest import Client
              client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
              msg = client.messages.create(
                  from_=os.environ["TWILIO_WHATSAPP_FROM"],
                  to=f"whatsapp:{kwargs['to']}",
                  body=kwargs["message"]
              )
              return ToolResult(success=True, data={"sid": msg.sid})
          except Exception as e:
              return ToolResult(success=False, error=str(e))

FILE: integrations/clients/calendar.py  (NEW)

  import os
  from datetime import datetime, timedelta
  from pathlib import Path
  from integrations.base import BaseIntegration, ToolResult

  CALENDAR_PATH = Path("memory/calendar.ics")

  class CalendarIntegration(BaseIntegration):
      name = "calendar"
      description = "Manage a local calendar (.ics file)"
      required_config = []   # no required env vars

      def is_available(self) -> bool:
          return True   # always available — uses stdlib + local file

      def get_tools(self) -> list:
          return [
              {
                  "name": "add_event",
                  "description": "Add an event to calendar",
                  "parameters": {
                      "title": {"type": "string"},
                      "date": {"type": "string", "description": "YYYY-MM-DD"},
                      "time": {"type": "string", "description": "HH:MM"},
                      "duration_minutes": {"type": "integer", "default": 60}
                  }
              },
              {
                  "name": "list_events",
                  "description": "List upcoming events",
                  "parameters": {
                      "days_ahead": {"type": "integer", "default": 7}
                  }
              }
          ]

      def execute(self, tool_name: str, **kwargs) -> ToolResult:
          CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
          try:
              if tool_name == "add_event":
                  return self._add_event(**kwargs)
              elif tool_name == "list_events":
                  return self._list_events(**kwargs)
              return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
          except Exception as e:
              return ToolResult(success=False, error=str(e))

      def _add_event(self, title, date, time="09:00", duration_minutes=60) -> ToolResult:
          dt_start = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
          dt_end = dt_start + timedelta(minutes=duration_minutes)
          uid = f"{dt_start.strftime('%Y%m%dT%H%M%S')}-jarvis"
          block = (
              f"BEGIN:VEVENT\n"
              f"DTSTART:{dt_start.strftime('%Y%m%dT%H%M%S')}\n"
              f"DTEND:{dt_end.strftime('%Y%m%dT%H%M%S')}\n"
              f"SUMMARY:{title}\n"
              f"UID:{uid}\n"
              f"END:VEVENT\n"
          )
          if not CALENDAR_PATH.exists():
              CALENDAR_PATH.write_text("BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n")
          content = CALENDAR_PATH.read_text()
          content = content.replace("END:VCALENDAR", block + "END:VCALENDAR")
          CALENDAR_PATH.write_text(content)
          return ToolResult(success=True, data={"event": title, "date": date, "time": time})

      def _list_events(self, days_ahead=7) -> ToolResult:
          if not CALENDAR_PATH.exists():
              return ToolResult(success=True, data={"events": []})
          content = CALENDAR_PATH.read_text()
          import re
          now = datetime.now()
          cutoff = now + timedelta(days=days_ahead)
          events = []
          for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", content, re.DOTALL):
              summary = re.search(r"SUMMARY:(.*)", block)
              dtstart = re.search(r"DTSTART:(.*)", block)
              if summary and dtstart:
                  try:
                      dt = datetime.strptime(dtstart.group(1).strip(), "%Y%m%dT%H%M%S")
                      if now <= dt <= cutoff:
                          events.append({"title": summary.group(1).strip(), "datetime": str(dt)})
                  except ValueError:
                      pass
          return ToolResult(success=True, data={"events": sorted(events, key=lambda x: x["datetime"])})

Wire integrations into existing system:

FILE: core/agent/controller.py  (MODIFY — MainController)
  In __init__, add:
    from integrations.loader import IntegrationLoader
    from integrations import registry as api_registry
    self.integration_loader = IntegrationLoader()
    result = self.integration_loader.load_all(self.config, api_registry)
    logger.info(f"Integrations: {result['loaded']} loaded, {len(result['skipped'])} skipped")

FILE: core/execution/dispatcher.py  (MODIFY)
  In execute() method, after checking TOOL_REGISTRY but before returning "not found":
    from integrations import registry as api_registry
    tool = api_registry.get_tool(tool_name)   # use actual method name from registry.py
    if tool is not None:
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: tool.execute(tool_name, **args)
        )

FILE: core/llm/task_planner.py  (MODIFY)
  In the method that calls Ollama (find _call_ollama or similar):
    from integrations.registry import api_registry
    integration_schemas = api_registry.list_schemas()   # use actual method name
    # Merge integration_schemas into the tool list sent to LLM
    # Add them to SYSTEM_TOOL_SCHEMA["tools"] list before building the prompt

config/settings.env  (NEW — template file, user fills in their own values)
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

════════════════════════════════════════════════
DO — SESSION 4
════════════════════════════════════════════════
- Read integrations/base.py and registry.py before writing loader
- Use ACTUAL method names from registry.py for register() and list_schemas()
- is_available() must NEVER raise — catch all exceptions, return False
- One bad plugin must never block other plugins from loading
- All credentials come from os.environ only — never hardcoded
- 10 second timeout on all SMTP/IMAP connections

════════════════════════════════════════════════
DO NOT — SESSION 4
════════════════════════════════════════════════
- Do NOT hardcode any email address, password, or API key in any .py file
- Do NOT let integrations import from core.agent — one-way dependency only
- Do NOT let a plugin crash the loader — catch per-plugin, log and skip
- Do NOT use any external email libraries — smtplib + imaplib (stdlib only)

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 4
════════════════════════════════════════════════
1.  config/settings.env                    (new — template)
2.  integrations/base.py                   (modify — add is_available, get_tools to ABC)
3.  integrations/loader.py                 (new)
4.  integrations/clients/__init__.py       (new, empty)
5.  integrations/clients/email.py          (new)
6.  integrations/clients/whatsapp.py       (new)
7.  integrations/clients/calendar.py       (new)
8.  core/agent/controller.py               (modify — add loader.load_all)
9.  core/execution/dispatcher.py           (modify — add integration routing)
10. core/llm/task_planner.py               (modify — merge integration tool schemas)

════════════════════════════════════════════════
VERIFICATION — SESSION 4
════════════════════════════════════════════════
python -c "from integrations.loader import IntegrationLoader; print('OK')"
python -c "from integrations.clients.email import EmailIntegration; e=EmailIntegration(); print('available:', e.is_available())"
python -c "from integrations.clients.calendar import CalendarIntegration; c=CalendarIntegration(); print('available:', c.is_available())"
python main.py --gui
# Check startup logs: should say "Integrations: [calendar] loaded, [email, whatsapp] skipped" (if env not set)
```

---
---

# SESSION 5 — STARTUP HEALTH CHECK + FINAL MAIN.PY WIRING
# ONE JOB: Health check on startup. Everything wired in main.py. Clean launch.
# Result: python main.py --gui launches everything, shows health status, GUI live.
# Prerequisite: Sessions 0–4 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0–4 are complete.
Python 3.11+, Windows 11, Ollama at http://localhost:11434.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
  core/introspection/health.py  — read HealthReport and HealthCheck — find actual method names
  core/agent/controller.py      — read MainController in current state
  core/controller_v2.py         — read JarvisControllerV2 in current state (all sessions applied)
  main.py                       — read the current state carefully before modifying

════════════════════════════════════════════════
SESSION 5 GOAL — HEALTH CHECK + MAIN.PY POLISH
════════════════════════════════════════════════

FILE: core/introspection/health.py  (MODIFY — keep all existing classes and methods)

Add this function at module level (not inside a class):

  def run_startup_health_check(controller=None) -> "HealthReport":
      """Run all checks and return a HealthReport. Print to terminal."""
      checks = {}

      # 1. Ollama reachable
      try:
          import urllib.request
          urllib.request.urlopen("http://localhost:11434", timeout=3)
          checks["ollama_reachable"] = True
      except Exception:
          checks["ollama_reachable"] = False

      # 2. ChromaDB installed
      try:
          import chromadb
          checks["chromadb_ready"] = True
      except ImportError:
          checks["chromadb_ready"] = None   # None = warn (optional)

      # 3. Memory SQLite accessible
      try:
          from pathlib import Path
          db = Path("memory/memory.db")
          checks["memory_sqlite"] = db.exists()
      except Exception:
          checks["memory_sqlite"] = False

      # 4. Voice deps installed
      try:
          import pvporcupine, sounddevice
          checks["voice_deps"] = True
      except ImportError:
          checks["voice_deps"] = None   # optional

      # 5. Config file exists
      from pathlib import Path
      checks["config_loaded"] = Path("config/jarvis.ini").exists()

      # 6. Integrations loaded (if controller available)
      if controller and hasattr(controller, "integration_loader"):
          try:
              result = getattr(controller, "_integration_result", {})
              n = len(result.get("loaded", []))
              checks["integrations"] = f"{n} loaded"
          except Exception:
              checks["integrations"] = "unknown"
      else:
          checks["integrations"] = "not wired"

      # Print to terminal with color (try colorama, fallback to plain)
      try:
          from colorama import Fore, Style, init
          init(autoreset=True)
          ok    = Fore.GREEN  + "✅"
          warn  = Fore.YELLOW + "⚠️"
          fail  = Fore.RED    + "❌"
          reset = Style.RESET_ALL
      except ImportError:
          ok = "OK"; warn = "WARN"; fail = "FAIL"; reset = ""

      print("\n═══ JARVIS STARTUP HEALTH ═══")
      for key, val in checks.items():
          if val is True or (isinstance(val, str) and val):
              icon = ok
          elif val is None:
              icon = warn
          else:
              icon = fail
          print(f"  {icon} {key}: {val}{reset}")
      print("══════════════════════════════\n")

      # Build and return HealthReport using existing class
      # Use whatever constructor/fields HealthReport actually has
      try:
          report = HealthReport(**{k: v for k, v in checks.items()
                                   if k in HealthReport.__dataclass_fields__})
      except Exception:
          report = HealthReport()   # fallback to empty report
      return report

FILE: main.py  (MODIFY — consolidate all flags added in Sessions 1-4, polish startup)

Read the current main.py carefully. Make sure these flags exist (add any missing):
  --voice      (store_true) — start voice mode
  --gui        (store_true) — start web dashboard at localhost:7070
  --dashboard  (store_true) — alias for --gui (backward compat)

The startup sequence must be in this order:
  1. Parse args
  2. Load config (config/jarvis.ini)
  3. Instantiate controller (JarvisControllerV2 or MainController — use actual)
  4. Run startup health check
  5. If --gui or --dashboard: start dashboard thread
  6. Push initial state to dashboard
  7. Start controller (run_cli / run — use actual method)

Add step 4 — startup health check — right after controller is instantiated:
  from core.introspection.health import run_startup_health_check
  health_report = run_startup_health_check(controller)

Add step 5 — GUI startup (consolidate the wiring from Session 1):
  if args.gui or getattr(args, "dashboard", False):
      import threading, uvicorn
      from dashboard.server import app as dashboard_app, set_controller, update_state
      set_controller(controller)
      try:
          update_state(
              session_id=getattr(controller, "session_id", "jarvis-1"),
              model=getattr(getattr(controller, "llm", None), "model", "unknown"),
              state="IDLE",
              ollama_online=getattr(health_report, "ollama_reachable", False)
          )
      except Exception:
          pass
      def _run_dashboard():
          uvicorn.run(dashboard_app, host="127.0.0.1", port=7070, log_level="warning")
      t = threading.Thread(target=_run_dashboard, daemon=True)
      t.start()
      print("Dashboard: http://localhost:7070")

Store integration result on controller for health check:
  In core/agent/controller.py __init__, after load_all():
    self._integration_result = result   # store for health check to read

════════════════════════════════════════════════
DO — SESSION 5
════════════════════════════════════════════════
- Health check runs every startup — lightweight, non-blocking
- health_report available so dashboard can show ollama_online status
- GUI thread is daemon=True — never blocks main exit
- colorama import wrapped in try/except — terminal color is optional
- Health check must complete in under 5 seconds total

════════════════════════════════════════════════
DO NOT — SESSION 5
════════════════════════════════════════════════
- Do NOT let health check failures crash startup — it's diagnostic only
- Do NOT block startup if Ollama is offline — warn and continue
- Do NOT duplicate the GUI thread if --gui and --dashboard both passed

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 5
════════════════════════════════════════════════
1. core/introspection/health.py   (modify — add run_startup_health_check)
2. core/agent/controller.py       (modify — store _integration_result)
3. main.py                        (modify — health check + all flags consolidated)

════════════════════════════════════════════════
VERIFICATION — SESSION 5
════════════════════════════════════════════════
python -c "from core.introspection.health import run_startup_health_check; print('OK')"
python main.py --help
# Must show: --voice, --gui, --dashboard flags

python main.py --gui
# Terminal should show:
# ═══ JARVIS STARTUP HEALTH ═══
#   ✅ ollama_reachable: True   (or ❌ if not running)
#   ⚠️  chromadb_ready: None
#   ...
# Dashboard: http://localhost:7070
# Open http://localhost:7070 → state badge should be IDLE (green)
```

---
---

# SESSION 6 — MULTI-MODEL ROUTING + PROACTIVE NOTIFICATIONS
# ONE JOB: Route tasks to best LLM. CPU/RAM alerts. Windows notifications.
# Result: Planner uses deepseek, chat uses mistral. System alerts when CPU > 90%.
# Prerequisite: Sessions 0–5 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0–5 are complete.
Python 3.11+, Windows 11, Ollama at http://localhost:11434.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
  core/llm/client.py         — find complete(), complete_json(), and __init__
  core/agent/controller.py   — find where LLM calls are made for planning vs chat
  core/controller_v2.py      — find process() and how it calls LLM

════════════════════════════════════════════════
INSTALL BEFORE STARTING
════════════════════════════════════════════════
pip install plyer psutil

════════════════════════════════════════════════
SESSION 6 GOAL — MULTI-MODEL + NOTIFICATIONS
════════════════════════════════════════════════

PART A: Multi-model routing

FILE: core/llm/model_router.py  (NEW)

  import time, logging
  from typing import Optional

  logger = logging.getLogger(__name__)

  DEFAULT_MODELS = {
      "planning":   "deepseek-r1:8b",
      "chat":       "mistral:7b",
      "vision":     "llava",
      "synthesis":  "deepseek-r1:8b",
      "embedding":  "nomic-embed-text",
      "fallback":   "mistral:7b",
  }

  class ModelRouter:
      def __init__(self, config=None):
          self._models = dict(DEFAULT_MODELS)
          if config and config.has_section("models"):
              for key in DEFAULT_MODELS:
                  opt = f"{key}_model" if key != "fallback" else "fallback_model"
                  if config.has_option("models", opt):
                      self._models[key] = config.get("models", opt)
          self._cache: dict = {}
          self._cache_time: float = 0.0
          self._cache_ttl: float = 60.0   # refresh every 60 seconds

      def route(self, task_type: str) -> str:
          return self._models.get(task_type, self._models["fallback"])

      def _refresh_cache(self) -> None:
          if time.time() - self._cache_time < self._cache_ttl:
              return
          try:
              import urllib.request, json
              with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
                  data = json.loads(r.read())
              available = {m["name"] for m in data.get("models", [])}
              self._cache = {name: (name in available) for name in self._models.values()}
              self._cache_time = time.time()
          except Exception as e:
              logger.debug(f"Model availability check failed: {e}")
              # Keep stale cache — do not clear it

      def is_available(self, model_name: str) -> bool:
          self._refresh_cache()
          return self._cache.get(model_name, False)

      def get_best_available(self, task_type: str) -> str:
          preferred = self.route(task_type)
          if self.is_available(preferred):
              return preferred
          fallback = self._models["fallback"]
          if self.is_available(fallback):
              logger.warning(f"Model {preferred} unavailable, using fallback {fallback}")
              return fallback
          # Last resort: return fallback even if not confirmed available
          return fallback

      def list_available(self) -> dict:
          self._refresh_cache()
          return dict(self._cache)

FILE: core/llm/client.py  (MODIFY — keep ALL existing code, add to it)

Add to __init__:
  self.model_router = None

Add method:
  def set_router(self, router) -> None:
      self.model_router = router

Modify complete() signature:
  def complete(self, prompt: str, system: str = "", temperature: float = 0.1,
               task_type: str = "chat") -> str:
  At the start of complete(), add:
      if self.model_router is not None:
          model_to_use = self.model_router.get_best_available(task_type)
      else:
          model_to_use = self.model
  Replace self.model in the API payload with model_to_use.
  Keep all existing error handling exactly as is.

Modify complete_json() to accept and pass task_type="planning":
  Add task_type param and pass it down to complete()

FILE: core/agent/controller.py  (MODIFY — MainController)
  In __init__, add:
    from core.llm.model_router import ModelRouter
    self.model_router = ModelRouter(config=self.config)
    self.llm.set_router(self.model_router)

  Find planning calls: pass task_type="planning"
  Find chat calls: pass task_type="chat"

  Also: update_state model field after router is ready:
    try:
        from dashboard.server import update_state
        update_state(model=self.model_router.route("chat"))
    except ImportError:
        pass

config/jarvis.ini — add this section (show updated complete file at end):
  [models]
  planning_model = deepseek-r1:8b
  chat_model = mistral:7b
  vision_model = llava
  synthesis_model = deepseek-r1:8b
  fallback_model = mistral:7b

PART B: Proactive notifications + resource monitor

FILE: core/proactive/__init__.py  (NEW — empty)

FILE: core/proactive/notifier.py  (NEW)

  import logging, time
  logger = logging.getLogger(__name__)

  class NotificationManager:
      def notify(self, message: str, level: str = "info", voice_layer=None) -> None:
          ts = time.strftime("%H:%M")
          print(f"\n[{ts}][JARVIS/{level.upper()}] {message}")
          try:
              from plyer import notification
              notification.notify(title="Jarvis", message=message[:256], timeout=5)
          except Exception:
              pass   # plyer optional
          if voice_layer is not None:
              try:
                  import asyncio
                  loop = asyncio.get_event_loop()
                  if loop.is_running():
                      loop.create_task(voice_layer.speak(message))
              except Exception:
                  pass

      def schedule_reminder(self, message: str, in_seconds: int) -> None:
          import asyncio
          try:
              loop = asyncio.get_running_loop()
              loop.create_task(self._delayed_notify(message, in_seconds))
          except RuntimeError:
              pass   # no running loop — skip

      async def _delayed_notify(self, message: str, delay: int) -> None:
          import asyncio
          await asyncio.sleep(delay)
          self.notify(message)

FILE: core/proactive/background_monitor.py  (NEW)

  import asyncio, logging
  logger = logging.getLogger(__name__)

  class BackgroundMonitor:
      def __init__(self, notifier, config=None):
          self.notifier = notifier
          self.cpu_threshold = 90
          self.ram_threshold = 90
          if config and config.has_section("proactive"):
              self.cpu_threshold = config.getint("proactive", "cpu_alert_threshold", fallback=90)
              self.ram_threshold = config.getint("proactive", "ram_alert_threshold", fallback=90)
          self._tasks: list = []
          self._running = False

      async def start(self) -> None:
          self._running = True
          self._tasks.append(asyncio.create_task(self._monitor_resources()))

      async def stop(self) -> None:
          self._running = False
          for t in self._tasks:
              t.cancel()
          self._tasks.clear()

      async def _monitor_resources(self) -> None:
          while self._running:
              await asyncio.sleep(60)
              try:
                  import psutil
                  cpu = psutil.cpu_percent(interval=1)
                  ram = psutil.virtual_memory().percent
                  if cpu > self.cpu_threshold:
                      self.notifier.notify(f"⚠️ CPU at {cpu:.0f}%", level="warn")
                  if ram > self.ram_threshold:
                      self.notifier.notify(f"⚠️ RAM at {ram:.0f}%", level="warn")
              except ImportError:
                  pass

Wire into controller_v2.py:
  In __init__, add:
    from core.proactive.notifier import NotificationManager
    from core.proactive.background_monitor import BackgroundMonitor
    self.notifier = NotificationManager()
    self.monitor = BackgroundMonitor(self.notifier, self.config)

  In _check_due_goals(): replace print() with self.notifier.notify()

  In run_cli() / start():
    asyncio.create_task(self.monitor.start())

  In shutdown():
    await self.monitor.stop()

config/jarvis.ini — also add:
  [proactive]
  cpu_alert_threshold = 90
  ram_alert_threshold = 90
  goal_check_interval_minutes = 5

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 6
════════════════════════════════════════════════
1. core/llm/model_router.py              (new)
2. core/llm/client.py                    (modify — set_router, task_type param)
3. core/proactive/__init__.py            (new, empty)
4. core/proactive/notifier.py            (new)
5. core/proactive/background_monitor.py  (new)
6. core/controller_v2.py                 (modify — wire notifier + monitor)
7. core/agent/controller.py              (modify — wire model_router)
8. config/jarvis.ini                     (show complete updated file)

════════════════════════════════════════════════
VERIFICATION — SESSION 6
════════════════════════════════════════════════
python -c "from core.llm.model_router import ModelRouter; r=ModelRouter(); print(r.route('chat'))"
python -c "from core.proactive.notifier import NotificationManager; print('OK')"
python main.py --gui
# Session info bar in dashboard should show current model name
```

---
---

# SESSION 7 — HARDWARE CONTROL + SCREEN/GUI AUTOMATION
# ONE JOB: Arduino serial control. Screen capture + OCR. Click/type automation.
# Result: Jarvis can read sensors, control hardware, and automate the desktop.
# Prerequisite: Sessions 0–6 complete.

```
You are an expert Python systems engineer.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0–6 are complete.
Python 3.11+, Windows 11.

════════════════════════════════════════════════
CRITICAL — READ THESE FILES BEFORE WRITING ANYTHING
════════════════════════════════════════════════
  core/hardware/serial_controller.py  — read the FULL file. It already has
                                         SerialController with simulation mode.
                                         EXTEND it — do NOT rewrite it.
  core/tools/builtin_tools.py         — find register_all_tools() and _assert_safe_path()
  core/autonomy/risk_evaluator.py     — find _DEFAULT_LOW and _DEFAULT_CONFIRM sets
  core/llm/task_planner.py            — find SYSTEM_TOOL_SCHEMA

════════════════════════════════════════════════
INSTALL BEFORE STARTING
════════════════════════════════════════════════
pip install pyserial pyautogui pillow pygetwindow pytesseract
# Also install Tesseract OCR binary for Windows:
# https://github.com/UB-Mannheim/tesseract/wiki

════════════════════════════════════════════════
SESSION 7 GOAL — HARDWARE + GUI TOOLS
════════════════════════════════════════════════

PART A: Hardware

FILE: core/hardware/serial_controller.py  (MODIFY — extend only, read first)

Add to existing SerialController class:
  async def async_send_command(self, cmd: str, value: str = "") -> dict:
      loop = asyncio.get_running_loop()
      result = await loop.run_in_executor(None, self.send_command, cmd, value)
      return {"success": True, "response": str(result), "simulated": self._simulation_mode}

  async def firmware_ping(self) -> bool:
      if self._simulation_mode:
          return True
      try:
          r = await self.async_send_command("PING")
          return "PONG" in str(r.get("response", ""))
      except Exception:
          return False

  async def sensor_read_loop(self, callback, interval: float = 1.0) -> None:
      import random
      while getattr(self, "_running", True):
          await asyncio.sleep(interval)
          if self._simulation_mode:
              data = {
                  "temperature": round(random.uniform(20.0, 25.0), 1),
                  "humidity": round(random.uniform(40.0, 60.0), 1),
                  "simulated": True
              }
          else:
              data = await self.async_send_command("READ_SENSORS")
          await callback(data) if asyncio.iscoroutinefunction(callback) else callback(data)

FILE: core/hardware/device_registry.py  (NEW)

  import json, logging
  from pathlib import Path
  logger = logging.getLogger(__name__)
  DEVICES_PATH = Path("config/devices.json")

  class DeviceRegistry:
      def __init__(self):
          self._devices: dict = {}
          self._instances: dict = {}
          self._load()

      def _load(self) -> None:
          if DEVICES_PATH.exists():
              try:
                  self._devices = json.loads(DEVICES_PATH.read_text())
              except Exception as e:
                  logger.warning(f"devices.json load failed: {e}")

      def _save(self) -> None:
          DEVICES_PATH.parent.mkdir(parents=True, exist_ok=True)
          tmp = DEVICES_PATH.with_suffix(".tmp")
          tmp.write_text(json.dumps(self._devices, indent=2))
          import os; os.replace(tmp, DEVICES_PATH)

      def register_device(self, name: str, com_port: str,
                          baud_rate: int = 115200, device_type: str = "arduino") -> None:
          self._devices[name] = {"com_port": com_port, "baud_rate": baud_rate,
                                  "device_type": device_type}
          self._save()

      def get_device(self, name: str):
          if name not in self._instances:
              if name not in self._devices:
                  raise KeyError(f"Device '{name}' not registered")
              cfg = self._devices[name]
              from core.hardware.serial_controller import SerialController
              sim = cfg["com_port"].upper() == "SIM"
              self._instances[name] = SerialController(
                  port=cfg["com_port"], baud_rate=cfg["baud_rate"],
                  simulation_mode=sim
              )
          return self._instances[name]

      def list_devices(self) -> list:
          result = []
          for name, cfg in self._devices.items():
              inst = self._instances.get(name)
              result.append({
                  "name": name,
                  "port": cfg["com_port"],
                  "device_type": cfg["device_type"],
                  "connected": inst is not None,
                  "simulation_mode": cfg["com_port"].upper() == "SIM"
              })
          return result

FILE: core/tools/hardware_tools.py  (NEW)

  from integrations.base import ToolResult
  from core.hardware.device_registry import DeviceRegistry
  _registry = DeviceRegistry()

  async def send_hardware_command(device_name: str, command: str, value: str = "") -> ToolResult:
      try:
          device = _registry.get_device(device_name)
          result = await device.async_send_command(command, value)
          return ToolResult(success=True, data=result)
      except Exception as e:
          return ToolResult(success=False, error=str(e))

  async def read_sensor(device_name: str, sensor_type: str = "all") -> ToolResult:
      try:
          device = _registry.get_device(device_name)
          result = await device.async_send_command("READ", sensor_type)
          return ToolResult(success=True, data=result)
      except Exception as e:
          return ToolResult(success=False, error=str(e))

  async def list_hardware_devices() -> ToolResult:
      return ToolResult(success=True, data={"devices": _registry.list_devices()})

  async def ping_device(device_name: str) -> ToolResult:
      try:
          device = _registry.get_device(device_name)
          alive = await device.firmware_ping()
          return ToolResult(success=True, data={"alive": alive})
      except Exception as e:
          return ToolResult(success=False, error=str(e))

PART B: Screen + GUI tools

FILE: core/tools/screen.py  (NEW)

  from pathlib import Path
  from datetime import datetime
  SCREENSHOT_DIR = Path("outputs/screenshots")

  def capture_screen() -> "ToolResult":
      try:
          import pyautogui
          from integrations.base import ToolResult
          SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
          ts = datetime.now().strftime("%Y%m%d_%H%M%S")
          path = SCREENSHOT_DIR / f"{ts}.png"
          img = pyautogui.screenshot()
          img.save(str(path))
          return ToolResult(success=True, data={"path": str(path), "width": img.width, "height": img.height})
      except ImportError:
          from integrations.base import ToolResult
          return ToolResult(success=False, error="pyautogui not installed")

  def capture_region(x: int, y: int, width: int, height: int) -> "ToolResult":
      try:
          import pyautogui
          from integrations.base import ToolResult
          SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
          ts = datetime.now().strftime("%Y%m%d_%H%M%S")
          path = SCREENSHOT_DIR / f"{ts}_region.png"
          img = pyautogui.screenshot(region=(x, y, width, height))
          img.save(str(path))
          return ToolResult(success=True, data={"path": str(path)})
      except ImportError:
          from integrations.base import ToolResult
          return ToolResult(success=False, error="pyautogui not installed")

  def find_text_on_screen(text: str) -> "ToolResult":
      try:
          import pyautogui, pytesseract
          from PIL import Image
          from integrations.base import ToolResult
          img = pyautogui.screenshot()
          data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
          matches = []
          for i, word in enumerate(data["text"]):
              if text.lower() in word.lower():
                  matches.append({"text": word, "x": data["left"][i], "y": data["top"][i],
                                   "w": data["width"][i], "h": data["height"][i]})
          return ToolResult(success=True, data={"matches": matches})
      except ImportError as e:
          from integrations.base import ToolResult
          return ToolResult(success=False, error=f"Missing dep: {e}")

  def describe_screen(llm_client=None) -> "ToolResult":
      try:
          import pyautogui
          from integrations.base import ToolResult
          img = pyautogui.screenshot()
          if llm_client is not None:
              # Try LLaVA if available
              try:
                  import base64, io
                  buf = io.BytesIO()
                  img.save(buf, format="PNG")
                  b64 = base64.b64encode(buf.getvalue()).decode()
                  desc = llm_client.complete(
                      prompt="Describe what you see on this screen briefly.",
                      images=[b64], task_type="vision"
                  )
                  return ToolResult(success=True, data={"description": desc})
              except Exception:
                  pass
          # Fallback: OCR dump
          try:
              import pytesseract
              text = pytesseract.image_to_string(img)
              return ToolResult(success=True, data={"ocr_text": text[:2000]})
          except ImportError:
              return ToolResult(success=True, data={"description": "Screen captured but no OCR available"})
      except ImportError:
          from integrations.base import ToolResult
          return ToolResult(success=False, error="pyautogui not installed")

FILE: core/tools/gui_control.py  (NEW)

  import asyncio, time
  from pathlib import Path
  from integrations.base import ToolResult

  GUI_AUDIT_DIR = Path("outputs/gui_audit")
  _FORBIDDEN_KEYWORDS = ("password", "passwd", "secret", "token", "apikey", "api_key")
  _LAST_CLICK_TIME = 0.0

  def _save_screenshot(label: str) -> str:
      try:
          import pyautogui
          GUI_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
          ts = int(time.time() * 1000)
          path = GUI_AUDIT_DIR / f"{ts}_{label}.png"
          pyautogui.screenshot().save(str(path))
          return str(path)
      except Exception:
          return ""

  def _validate_coords(x: int, y: int) -> bool:
      try:
          import pyautogui
          w, h = pyautogui.size()
          return 0 <= x < w and 0 <= y < h
      except Exception:
          return True   # can't check — allow

  async def click(x: int, y: int, button: str = "left") -> ToolResult:
      global _LAST_CLICK_TIME
      try:
          import pyautogui
      except ImportError:
          return ToolResult(success=False, error="pyautogui not installed")
      if not _validate_coords(x, y):
          return ToolResult(success=False, error=f"Coordinates ({x},{y}) outside screen bounds")
      _save_screenshot("before_click")
      await asyncio.sleep(0.3)   # safety delay — always, non-negotiable
      _LAST_CLICK_TIME = time.time()
      pyautogui.click(x, y, button=button)
      _save_screenshot("after_click")
      return ToolResult(success=True, data={"action": "click", "x": x, "y": y})

  async def double_click(x: int, y: int) -> ToolResult:
      try:
          import pyautogui
      except ImportError:
          return ToolResult(success=False, error="pyautogui not installed")
      if not _validate_coords(x, y):
          return ToolResult(success=False, error="Coordinates outside screen bounds")
      _save_screenshot("before_dblclick")
      await asyncio.sleep(0.3)
      pyautogui.doubleClick(x, y)
      _save_screenshot("after_dblclick")
      return ToolResult(success=True, data={"action": "double_click", "x": x, "y": y})

  async def right_click(x: int, y: int) -> ToolResult:
      try:
          import pyautogui
      except ImportError:
          return ToolResult(success=False, error="pyautogui not installed")
      if not _validate_coords(x, y):
          return ToolResult(success=False, error="Coordinates outside screen bounds")
      _save_screenshot("before_rightclick")
      await asyncio.sleep(0.3)
      pyautogui.rightClick(x, y)
      return ToolResult(success=True, data={"action": "right_click", "x": x, "y": y})

  async def type_text(text: str, interval: float = 0.05) -> ToolResult:
      # HARD SAFETY RULE: refuse if text contains sensitive keywords
      lower = text.lower()
      for kw in _FORBIDDEN_KEYWORDS:
          if kw in lower:
              return ToolResult(success=False, error=f"Refused: text contains '{kw}'")
      try:
          import pyautogui
      except ImportError:
          return ToolResult(success=False, error="pyautogui not installed")
      await asyncio.sleep(0.3)
      pyautogui.typewrite(text, interval=interval)
      return ToolResult(success=True, data={"action": "type_text", "length": len(text)})

  async def hotkey(*keys) -> ToolResult:
      try:
          import pyautogui
          await asyncio.sleep(0.1)
          pyautogui.hotkey(*keys)
          return ToolResult(success=True, data={"action": "hotkey", "keys": list(keys)})
      except ImportError:
          return ToolResult(success=False, error="pyautogui not installed")

  def get_active_window() -> ToolResult:
      try:
          import pygetwindow as gw
          win = gw.getActiveWindow()
          if win is None:
              return ToolResult(success=True, data={"title": None})
          return ToolResult(success=True, data={
              "title": win.title, "x": win.left, "y": win.top,
              "width": win.width, "height": win.height
          })
      except ImportError:
          return ToolResult(success=False, error="pygetwindow not installed")

Wire into existing files:

FILE: core/tools/builtin_tools.py  (MODIFY)
  In register_all_tools(), add registrations for:
    Hardware tools: send_hardware_command, read_sensor, list_hardware_devices, ping_device
    Screen tools:   capture_screen, capture_region, find_text_on_screen, describe_screen
    GUI tools:      click, double_click, right_click, type_text, hotkey, get_active_window

FILE: core/autonomy/risk_evaluator.py  (MODIFY)
  Add to _DEFAULT_LOW set:
    "capture_screen", "capture_region", "find_text_on_screen",
    "describe_screen", "get_active_window",
    "list_hardware_devices", "ping_device", "read_sensor"
  Add to _DEFAULT_CONFIRM set:
    "click", "double_click", "right_click", "type_text", "hotkey",
    "send_hardware_command"

FILE: core/llm/task_planner.py  (MODIFY)
  Add hardware and GUI tool schemas to SYSTEM_TOOL_SCHEMA tools list.
  Format must match existing schema entries exactly.

════════════════════════════════════════════════
SAFETY RULES — NON-NEGOTIABLE (hard-coded, not configurable)
════════════════════════════════════════════════
- 300ms sleep before every click — never remove this
- Save before + after screenshots for every click action
- Validate x,y within screen bounds before clicking
- type_text REFUSES if text contains: password, passwd, secret, token, apikey
- Never allow hardware commands at autonomy_level < 2 (check in risk_evaluator)

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 7
════════════════════════════════════════════════
1. core/hardware/serial_controller.py    (modify — extend with async methods)
2. core/hardware/device_registry.py      (new)
3. core/tools/hardware_tools.py          (new)
4. core/tools/screen.py                  (new)
5. core/tools/gui_control.py             (new)
6. core/tools/builtin_tools.py           (modify — register new tools)
7. core/autonomy/risk_evaluator.py       (modify — add to risk sets)
8. core/llm/task_planner.py              (modify — add tool schemas)

════════════════════════════════════════════════
VERIFICATION — SESSION 7
════════════════════════════════════════════════
python -c "from core.hardware.device_registry import DeviceRegistry; print('OK')"
python -c "from core.tools.screen import capture_screen; print('OK')"
python -c "from core.tools.gui_control import click; print('OK')"
python -c "
from core.tools.hardware_tools import list_hardware_devices
import asyncio
result = asyncio.run(list_hardware_devices())
print(result)
"
```

---
---

# SESSION 8 — SECURITY HARDENING + FULL TEST SUITE
# ONE JOB: Harden file access, add rate limiting, write all tests.
# Result: bandit shows no HIGH findings. pytest passes fully offline.
# Prerequisite: Sessions 0–7 complete.

```
You are an expert Python systems engineer and security auditor.
Project: Jarvis at D:\AI\Jarvis\
Sessions 0–7 are complete.
Python 3.11+, Windows 11.

════════════════════════════════════════════════
INSTALL BEFORE STARTING
════════════════════════════════════════════════
pip install pytest pytest-asyncio pytest-cov bandit

════════════════════════════════════════════════
SESSION 8 GOAL — SECURITY + TESTS
════════════════════════════════════════════════

PART A: Security hardening

FILE: core/tools/builtin_tools.py  (MODIFY)
  Find _assert_safe_path() (or equivalent path validation function). Add:
    from pathlib import Path
    resolved = Path(user_path).resolve()
    sandbox = Path("D:/AI/Jarvis").resolve()   # project root
    if ".." in str(Path(user_path)):
        raise PermissionError(f"Path traversal blocked: {user_path}")
    if not str(resolved).startswith(str(sandbox)):
        raise PermissionError(f"Path outside sandbox: {resolved}")
    if resolved.is_symlink():
        link_target = resolved.resolve()
        if not str(link_target).startswith(str(sandbox)):
            raise PermissionError(f"Symlink escapes sandbox: {link_target}")

  In read_file() (find it):
    import os
    size = os.path.getsize(path)
    if size > 10 * 1024 * 1024:   # 10MB
        raise ValueError(f"File too large: {size} bytes (max 10MB)")

FILE: core/execution/dispatcher.py  (MODIFY)
  Add helper method:
    def _sanitize_args(self, args: dict) -> dict:
        sanitized = {}
        for key, val in args.items():
            if isinstance(val, str):
                val = val.replace("\x00", "")   # strip null bytes
                if len(val) > 4096:
                    val = val[:4096]            # truncate oversized
            sanitized[key] = val
        return sanitized

  Call _sanitize_args(args) before every tool execution.

  Add rate limiting fields to __init__:
    self._call_count: int = 0
    self._call_window_start: float = time.time()
    self.MAX_CALLS_PER_MINUTE: int = 30

  In execute(), BEFORE dispatching to any tool:
    import time
    now = time.time()
    if now - self._call_window_start > 60:
        self._call_count = 0
        self._call_window_start = now
    self._call_count += 1
    if self._call_count > self.MAX_CALLS_PER_MINUTE:
        return ToolResult(success=False, error="Rate limit exceeded: 30 tool calls/minute")

FILE: audit/audit_logger.py  (MODIFY)
  Add log rotation:
    import os, gzip, shutil
    In write() (or whatever the log-write method is), before writing:
        if os.path.exists(self._log_path) and os.path.getsize(self._log_path) > 50 * 1024 * 1024:
            ts = int(time.time())
            gz_path = f"{self._log_path}.{ts}.gz"
            with open(self._log_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(self._log_path)
            # New empty log file created on next write

  Add secret scrubber:
    import re
    _SECRET_PATTERNS = [
        r'(?i)(password|passwd|pwd|token|secret|sid|api_key|apikey)\s*[=:]\s*\S+',
        r'[A-Za-z0-9]{32,}',   # long random strings (tokens)
    ]
    def scrub_secrets(text: str) -> str:
        for pattern in _SECRET_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text)
        return text

  Call scrub_secrets() on ALL text content before writing to log.

PART B: Test suite

All tests use pytest. All async tests use @pytest.mark.asyncio.
All external calls (Ollama, SMTP, hardware, APIs) MUST be mocked.
Tests must pass fully offline with no real hardware or internet.

FILE: tests/conftest.py  (REPLACE existing or create if missing)
  import pytest
  from unittest.mock import MagicMock, AsyncMock
  from configparser import ConfigParser
  from pathlib import Path

  @pytest.fixture
  def tmp_dir(tmp_path):
      return tmp_path

  @pytest.fixture
  def mock_config():
      cfg = ConfigParser()
      cfg["agent"] = {"max_iterations": "10"}
      cfg["models"] = {"chat_model": "mistral:7b", "fallback_model": "mistral:7b"}
      cfg["proactive"] = {"cpu_alert_threshold": "90"}
      return cfg

  @pytest.fixture
  def mock_llm():
      llm = MagicMock()
      llm.complete = MagicMock(return_value='{"communication_style": {"value": "casual", "confidence": 0.8}}')
      llm.complete_json = MagicMock(return_value={})
      return llm

  @pytest.fixture
  def mock_controller():
      ctrl = MagicMock()
      ctrl.process = MagicMock(return_value="test response")
      ctrl.session_id = "test-session"
      return ctrl

FILE: tests/test_risk_evaluator.py
  from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel
  Test:
    LOW tools are not blocked (read_file returns LOW)
    CONFIRM tools require confirmation (write_file returns CONFIRM)
    CRITICAL tools are hard-blocked
    Unknown tool names default to HIGH
    evaluate_plan() with mixed actions returns the highest risk level
    Risk ordering: LOW < CONFIRM < HIGH < CRITICAL
    Empty action list returns LOW or no error

FILE: tests/test_memory.py
  Mock sqlite3 and chromadb. Test:
    store_preference() saves and retrieve_preference() returns correct value
    store_conversation() and recall() work correctly
    HybridMemory initializes correctly with chromadb missing (graceful fallback)
    store_code_file() handles SyntaxError gracefully

FILE: tests/test_agent_loop.py
  from unittest.mock import MagicMock, patch, AsyncMock
  Test:
    Full loop run with mock planner returns ExecutionTrace
    Interrupt flag stops loop immediately
    max_iterations=2 stops loop after 2 iterations
    CRITICAL risk action → loop returns with risk_threshold_exceeded reason
    think_blocks populated from <think>content</think> in LLM response
    Long observation string truncated to 800 chars by _truncate_obs()
    Successful tool → trace.success = True
    Failed tool → trace.success = False (if that's how trace works)

FILE: tests/test_integrations.py
  from integrations.loader import IntegrationLoader
  from integrations.clients.email import EmailIntegration
  from integrations.clients.calendar import CalendarIntegration
  from integrations.clients.whatsapp import WhatsAppIntegration
  Test:
    IntegrationLoader.load_all() scans clients/ and returns dict with loaded/skipped keys
    Bad plugin (raises on import) does NOT crash loader — is in skipped list
    EmailIntegration.is_available() returns False when EMAIL_ADDRESS env var not set
    WhatsAppIntegration.is_available() returns False when twilio not installed
    CalendarIntegration.is_available() always returns True
    CalendarIntegration.execute("add_event") creates entry in calendar.ics (use tmp_path)
    ToolResult has success, data, error attributes

FILE: tests/test_profile.py
  from core.profile import UserProfileEngine
  from core.synthesis import ProfileSynthesizer
  Test:
    UserProfileEngine() loads defaults when memory/user_profile.json missing
    save() creates file — subsequent load() reads same data back
    save() is atomic — uses .tmp file (check tmp file gone after save)
    update_from_conversation() increments interaction_count
    "my name is Alice" in user_text → profile.name == "Alice"
    get_communication_style() returns correct string for each style value
    get_system_prompt_injection() result is under 300 characters
    ProfileSynthesizer.should_run() returns True at count=20, False at count=19
    synthesize() handles LLM returning invalid JSON → returns error key, does not raise
    apply_delta() only updates fields where confidence > 0.6

FILE: tests/test_security.py
  from core.tools.builtin_tools import _assert_safe_path (or equivalent)
  from core.execution.dispatcher import Dispatcher (or equivalent)
  from core.tools.gui_control import type_text
  import asyncio
  Test:
    "../../etc/passwd" path is blocked by path validation
    Symlink pointing outside sandbox is blocked
    File larger than 10MB raises ValueError in read_file
    Null bytes \x00 stripped from dispatcher args by _sanitize_args
    30 tool calls in under 1 minute triggers rate limit on 31st call
    type_text("my password is 123") returns ToolResult with success=False
    type_text("hello world") returns ToolResult with success=True
    scrub_secrets("token=abc123...") → "[REDACTED]" in output

════════════════════════════════════════════════
DO — SESSION 8
════════════════════════════════════════════════
- All tests mock every external dependency — tests pass with no internet
- Run bandit -r core/ -ll after security changes — target: zero HIGH findings
- Log rotation: gzip completed synchronously before new log starts
- Rate limit resets every 60 seconds per dispatcher instance

════════════════════════════════════════════════
DO NOT — SESSION 8
════════════════════════════════════════════════
- Do NOT use unittest — pytest only
- Do NOT skip testing error paths and failure cases
- Do NOT use shell=True in any subprocess call anywhere in the codebase
- Do NOT allow any test to make real network calls or touch real hardware

════════════════════════════════════════════════
FILE DELIVERY ORDER — SESSION 8
════════════════════════════════════════════════
 1. core/tools/builtin_tools.py        (modify — path traversal + file size)
 2. core/execution/dispatcher.py       (modify — _sanitize_args + rate limit)
 3. audit/audit_logger.py              (modify — log rotation + scrub_secrets)
 4. tests/conftest.py                  (replace/create)
 5. tests/test_risk_evaluator.py       (new)
 6. tests/test_memory.py               (new)
 7. tests/test_agent_loop.py           (new)
 8. tests/test_integrations.py         (new)
 9. tests/test_profile.py              (new)
10. tests/test_security.py             (new)

════════════════════════════════════════════════
VERIFICATION — SESSION 8
════════════════════════════════════════════════
pytest tests/ -v --tb=short -x
# All tests must pass with no real hardware or internet

pytest tests/ --cov=core --cov-report=term-missing
# Target: 70%+ coverage across core/

bandit -r core/ -ll
# Target: zero HIGH severity findings

python main.py --gui
# Full system should still run after security changes
```

---
---

## COMPLETE SESSION SUMMARY

| Session | One Job | Max Files | Key Deliverable |
|---------|---------|-----------|-----------------|
| **0** | Build GUI dashboard module | 8 | http://localhost:7070 dark HUD, live state badge |
| **1** | Wire voice into controller | 2 | `--voice` flag works, `--gui` shows live state |
| **2** | Wire goals + scheduler + confidence | 2 | "remind me to X" → goal saved, confidence gates tools |
| **3** | User profile + synthesis | 3 | Jarvis learns your name, style, adapts after 20 turns |
| **4** | Integration plugins (email/cal/WA) | 10 | Email send/read, calendar add/list, WhatsApp send |
| **5** | Health check + main.py polish | 3 | Startup shows ✅/⚠️/❌ for each component |
| **6** | Multi-model routing + notifications | 8 | Planner→deepseek, chat→mistral, CPU/RAM alerts |
| **7** | Hardware + screen/GUI automation | 8 | Arduino serial, screen OCR, click/type automation |
| **8** | Security hardening + full tests | 10 | No path traversal, rate limiting, pytest passes offline |

**Run in strict order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**

---

## GUI CUSTOMIZATION GUIDE

```
RETHEME ENTIRE UI (3 line change):
  Edit dashboard/static/style.css, change:
    --bg:     #0d0d1a    ← main background
    --accent: #4ecca3    ← highlight color
    --bg2:    #12122a    ← card/panel color
  Everything updates. No rebuild. No restart needed (hard refresh browser).

ADD A NEW PAGE:
  1. Create dashboard/templates/newpage.html ({% extends "base.html" %})
  2. Add GET /newpage route in dashboard/server.py
  3. Nav link auto-appears if added to base.html nav
  Zero changes to core/ needed.

PUSH LIVE DATA FROM CORE:
  try:
      from dashboard.server import update_state
      update_state(state="THINKING")
      update_state(last_response="Done.", memory_count=42)
  except ImportError:
      pass   # GUI not running — always silently skip

REPLACE THE ENTIRE FRONTEND:
  Swap dashboard/templates/ and dashboard/static/ with any design.
  The WebSocket endpoint ws://localhost:7070/ws always pushes the same JSON.
  Any frontend — plain HTML, Svelte, React — just connect to the WS.
```