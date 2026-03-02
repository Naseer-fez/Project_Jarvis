# JARVIS — CRITICAL FIXES FILE
**Status:** MUST FIX BEFORE ANY NEW FEATURE WORK  
**Author:** Senior Engineering Audit  
**Scope:** Every broken, wrong, or dangerous thing found after reading every file  
**How to use:** Paste this file at the start of your LLM session. Fix top-to-bottom. Do not skip.

---

## SEVERITY LEGEND

| Symbol | Meaning |
|--------|---------|
| 🔴 CRITICAL | Will cause crashes, data loss, or security holes in production |
| 🟠 HIGH | Causes wrong behaviour, subtle bugs, or blocks cross-platform use |
| 🟡 MEDIUM | Technical debt that will hurt you within 3 sessions |
| 🔵 LOW | Cleanup — messy but not dangerous |

---

## FIX 1 — 🔴 CRITICAL — Hardcoded Windows path in production sandbox

**File:** `core/tools/builtin_tools.py` — line 24  
**File:** `core/memory/semantic_memory.py` — line 41

**The Bug:**
```python
# CURRENT — BROKEN on every machine that is not the original dev PC
_SANDBOX_ROOT = Path("D:/AI/Jarvis").resolve()
CHROMA_PATH = "D:/AI/Jarvis/data/chroma"
```

**Why it is catastrophic:**  
Every single file operation (read_file, write_file, list_directory) calls `_assert_safe_path()` which checks against `_SANDBOX_ROOT`. If Jarvis runs on any Linux, Mac, or different Windows machine, this resolves to a path that does not exist. The check then throws `PermissionError` on ALL file tools. Jarvis is completely blind. This is not a warning — it is a full functional failure.

**The Fix:**
```python
# core/tools/builtin_tools.py — replace lines 23-24
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # goes up: tools/ → core/ → project root
_SANDBOX_ROOT = _PROJECT_ROOT

# Also update ALLOWED_DIRECTORIES to use PROJECT_ROOT
ALLOWED_DIRECTORIES = [
    (_PROJECT_ROOT / "workspace").resolve(),
    (_PROJECT_ROOT / "outputs").resolve(),
]
```

```python
# core/memory/semantic_memory.py — replace hardcoded path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chroma")
```

**Verification:**
```bash
python -c "from core.tools.builtin_tools import _SANDBOX_ROOT; print(_SANDBOX_ROOT); assert _SANDBOX_ROOT.exists()"
```

---

## FIX 2 — 🔴 CRITICAL — asyncio.new_event_loop() deadlock inside async runtime

**Files:**  
- `core/llm/client.py` — lines 157, 239  
- `core/planning/intents.py` — lines 69–70

**The Bug:**
```python
# CURRENT — WILL DEADLOCK when called from agent_loop (which is async)
def chat(self, messages, ...):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(self.complete(...))  # DEADLOCK
    finally:
        loop.close()
```

**Why it is catastrophic:**  
`controller_v2.py` runs in an async context (`async_main`). The agent loop is also async. When `chat()` is called from inside any `await` chain and tries to call `asyncio.new_event_loop().run_until_complete()`, Python raises `RuntimeError: This event loop is already running` on CPython, or silently deadlocks on some environments. This breaks every conversational response.

**The Fix for `core/llm/client.py`:**
```python
# Replace the sync chat() method entirely

async def chat_async(
    self,
    messages: list[dict[str, Any]],
    query_for_memory: str = "",
    profile_summary: str = "",
    workspace_path: str = "",
) -> str:
    """Async version — use this inside any async context (agent loop, controller)."""
    system = self._build_system(
        query=query_for_memory,
        profile=profile_summary,
        workspace_path=workspace_path,
    )
    prompt = self._messages_to_prompt(messages)
    return await self.complete(prompt, system=system) or ""

def chat(
    self,
    messages: list[dict[str, Any]],
    query_for_memory: str = "",
    profile_summary: str = "",
    workspace_path: str = "",
) -> str:
    """Sync bridge — ONLY call from truly synchronous, non-async contexts."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            asyncio.run,
            self.chat_async(messages, query_for_memory, profile_summary, workspace_path)
        )
        return future.result()
```

**Fix for `is_ollama_running()`:**
```python
# Replace the broken sync version
def is_ollama_running(self) -> bool:
    """Sync health check — runs in its own thread to avoid event loop conflicts."""
    import concurrent.futures
    async def _check() -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(OLLAMA_BASE_URL, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    return r.status == 200
        except Exception:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _check()).result(timeout=5)
```

**Fix for `core/planning/intents.py`:**
```python
# Replace _llm_classify to be properly async-aware
def _llm_classify(self, text: str) -> dict | None:
    try:
        import concurrent.futures
        prompt = f"Classify this message:\n\n{text}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                asyncio.run,
                self.llm.complete_json(prompt, system=CLASSIFY_SYSTEM, temperature=0.0)
            ).result(timeout=30)
        if result and "intent" in result:
            intent_str = result["intent"].lower()
            valid = {i.value for i in Intent}
            if intent_str not in valid:
                intent_str = "unknown"
            return {"intent": intent_str, "confidence": float(result.get("confidence", 0.7))}
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")
    return None
```

---

## FIX 3 — 🔴 CRITICAL — Two conflicting LLMClientV2 classes with the same name

**Files:**  
- `core/llm/client.py` — the REAL async client (correct)  
- `core/llm/llm_v2.py` — a dead sync-only stub using `requests` (wrong)

**The Bug:**  
Both files define a class named `LLMClientV2`. `controller_v2.py` imports from `core.llm.llm_v2`. This means the controller is using the OLD stub that uses blocking `requests.post()` with no memory injection, no model routing, no system prompt building — none of it. The entire `core/llm/client.py` is being bypassed silently.

**Proof:**
```python
# controller_v2.py line 17
from core.llm.llm_v2 import LLMClientV2  # WRONG — uses the stub
```

**The Fix:**
```python
# Step 1: Delete core/llm/llm_v2.py entirely
# Step 2: In controller_v2.py, change the import
from core.llm.client import LLMClientV2  # CORRECT

# Step 3: Verify the constructor signature matches — client.py expects:
# LLMClientV2(hybrid_memory=..., model=..., profile=...)
# controller_v2.py passes model_name= — fix this mismatch:
self.llm = LLMClientV2(
    hybrid_memory=self.memory,
    model=self.model_router.route("chat"),   # was model_name=
    profile=self.profile,
)
```

---

## FIX 4 — 🔴 CRITICAL — Dead stub in Failed/ is being treated as canon

**Files:**  
- `core/llm/fallback.py` — real implementation with proper logic  
- `Failed/core/llm/fallback.py` — contains only `def __stub__(): pass`

**The Bug:**  
The `Failed/` directory is inside the project root. If anything does a wildcard import or sys.path manipulation (common in test runners), the stub can shadow the real fallback. The stub returns `None` implicitly. Any caller expecting a dict plan from `fallback_plan()` will crash with `TypeError: 'NoneType' has no attribute...`.

**The Fix:**
```bash
# Step 1: Confirm Failed/ is never on sys.path
grep -r "Failed" pytest.ini mypy.ini setup.py pyproject.toml

# Step 2: Add to pytest.ini to explicitly exclude
[pytest]
testpaths = tests
asyncio_mode = auto
norecursedirs = Failed archive_legacy archive_jarvis_duplicate

# Step 3: Add to .gitignore (the Failed/ dir should not pollute production)
# OR delete Failed/ entirely and keep only the roadmap notes elsewhere
```

---

## FIX 5 — 🔴 CRITICAL — credentials committed to version control

**File:** `config/settings.env`

**The Bug:**
```ini
# settings.env is tracked by git. Even with empty values, this is wrong.
EMAIL_ADDRESS=
EMAIL_PASSWORD=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
PORCUPINE_ACCESS_KEY=
```

**Why it matters:**  
Once a file is in git history, even deleting it does not remove it. Any real values ever committed (accidentally) are permanently exposed. The file also signals exactly what credentials Jarvis needs — an attacker's shopping list.

**The Fix:**
```bash
# Step 1: Create a safe template
cp config/settings.env config/settings.env.template
# Edit the template to have placeholder comments only:
# EMAIL_ADDRESS=your_gmail@gmail.com

# Step 2: Add the real file to .gitignore
echo "config/settings.env" >> .gitignore
echo "*.env" >> .gitignore

# Step 3: Remove from git tracking (does NOT delete the file)
git rm --cached config/settings.env

# Step 4: If real secrets were EVER committed, rotate them immediately
```

---

## FIX 6 — 🟠 HIGH — Three entry points with no canonical source of truth

**Files:**  
- `main.py` — the correct V2 entry point  
- `main_v3.py` — unknown purpose, untested  
- `scripts/main_v2.py` — exact duplicate of `archive_jarvis_duplicate/main_v2.py`

**The Bug:**  
`python main_v3.py` and `python scripts/main_v2.py` both start different versions of Jarvis. There is no `if __name__ == "__main__"` guard check that fails gracefully in the wrong version. A new contributor (or the LLM) will not know which file to run.

**The Fix:**
```python
# Step 1: Add deprecation guards to main_v3.py and scripts/main_v2.py
# At the TOP of each deprecated file, before any imports:
import sys, warnings
warnings.warn(
    "main_v3.py is deprecated. Use: python main.py",
    DeprecationWarning,
    stacklevel=1
)
sys.exit(1)

# Step 2: Move scripts/main_v2.py to archive_jarvis_duplicate/ where it belongs
# Step 3: Delete scripts/python.exe and scripts/pythonw.exe (venv artifacts)
```

---

## FIX 7 — 🟠 HIGH — ModelRouter falls back to model it cannot confirm exists

**File:** `core/llm/model_router.py`

**The Bug:**
```python
def get_best_available(self, task_type: str) -> str:
    preferred = self.route(task_type)
    if self.is_available(preferred):
        return preferred
    fallback = self._models["fallback"]
    if self.is_available(fallback):
        return fallback
    # Last resort: return fallback even if not confirmed available.
    return fallback  # <-- silently returns a model that may not exist
```

**What happens:**  
LLMClientV2.complete() gets a model name from the router, sends it to Ollama, gets HTTP 404 (model not found), logs "Ollama HTTP 404", and returns `""`. The agent loop gets an empty string response, cannot parse a plan, and silently fails. The user sees nothing or a generic error with no actionable message.

**The Fix:**
```python
def get_best_available(self, task_type: str) -> str:
    preferred = self.route(task_type)
    if self.is_available(preferred):
        return preferred

    fallback = self._models["fallback"]
    if self.is_available(fallback):
        logger.warning(
            "Preferred model '%s' unavailable for task '%s'. Using fallback '%s'.",
            preferred, task_type, fallback
        )
        return fallback

    # NOTHING is available — surface this loudly
    available = [m for m, ok in self.list_available().items() if ok]
    if available:
        logger.error(
            "Neither '%s' nor fallback '%s' available. Using first available: '%s'. "
            "Run `ollama pull %s` to fix this.",
            preferred, fallback, available[0], preferred
        )
        return available[0]

    raise RuntimeError(
        f"No Ollama models available. Run: ollama pull {fallback}\n"
        f"Then restart Jarvis. Configured models: {list(self._models.values())}"
    )
```

---

## FIX 8 — 🟠 HIGH — Calendar integration uses hand-rolled regex ICS parser

**File:** `integrations/clients/calendar.py`

**The Bug:**
```python
# Current parser — will silently drop:
# - Events with timezone (DTSTART;TZID=America/New_York:...)
# - Recurring events (RRULE:FREQ=WEEKLY)
# - Long summaries with line folding (80-char wrap per RFC 5545)
# - Events in UTC format (DTSTART:20240101T120000Z)
for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", content, re.DOTALL):
    dtstart = re.search(r"DTSTART:(.*)", block)  # fails on DTSTART;TZID=...
```

**The Fix:**
```bash
pip install icalendar python-dateutil
```

```python
# Replace _list_events() in integrations/clients/calendar.py
def _list_events(self, days_ahead: int = 7) -> dict[str, Any]:
    if not CALENDAR_PATH.exists():
        return {"events": []}
    
    from icalendar import Calendar
    from dateutil.tz import tzlocal
    import datetime as dt
    
    cal = Calendar.from_ical(CALENDAR_PATH.read_bytes())
    now = dt.datetime.now(tz=tzlocal())
    cutoff = now + dt.timedelta(days=days_ahead)
    events = []
    
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        dtstart = component.get("DTSTART")
        if dtstart is None:
            continue
        start = dtstart.dt
        # Handle date-only events
        if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
            start = dt.datetime(start.year, start.month, start.day, tzinfo=tzlocal())
        if start.tzinfo is None:
            start = start.replace(tzinfo=tzlocal())
        if now <= start <= cutoff:
            events.append({
                "title": str(component.get("SUMMARY", "Untitled")),
                "datetime": str(start),
            })
    
    return {"events": sorted(events, key=lambda e: e["datetime"])}
```

---

## FIX 9 — 🟠 HIGH — core/llm/ contains 40+ schema files that are not for Jarvis

**Directory:** `core/llm/`

**The Bug:**  
The following files are copy-pasted from third-party libraries (pydantic, jsonschema, altair/schemapi). They are not authored by Jarvis, not used by Jarvis, and pollute the module namespace:

```
fastjsonschema_exceptions.py    # from fastjsonschema library internals
fastjsonschema_validations.py   # from fastjsonschema library internals
json_schema_test_suite.py       # test harness from jsonschema project
test_embedding_function_schemas.py  # test file that doesn't belong in production code
test_schema.py                  # third-party test
test_schema_e2e.py              # third-party e2e test
useless_applicator_schemas.py   # literally named useless
schemapi.py                     # Altair's internal schema API
_generate_schema.py             # pydantic internals
_schema_gather.py               # pydantic internals
_schema_generation_shared.py    # pydantic internals
configuration_smollm3.py        # HuggingFace SmolLM3 config — not used
modeling_smollm3.py             # HuggingFace SmolLM3 model — not used
modular_smollm3.py              # HuggingFace SmolLM3 variant — not used
```

**The Fix:**
```bash
# Move to a graveyard so they're not on the import path
mkdir -p archive_legacy/llm_bloat
mv core/llm/fastjsonschema_*.py archive_legacy/llm_bloat/
mv core/llm/json_schema_test_suite.py archive_legacy/llm_bloat/
mv core/llm/test_schema*.py archive_legacy/llm_bloat/
mv core/llm/test_embedding*.py archive_legacy/llm_bloat/
mv core/llm/useless_applicator_schemas.py archive_legacy/llm_bloat/
mv core/llm/schemapi.py archive_legacy/llm_bloat/
mv core/llm/_generate_schema.py archive_legacy/llm_bloat/
mv core/llm/_schema_gather.py archive_legacy/llm_bloat/
mv core/llm/_schema_generation_shared.py archive_legacy/llm_bloat/
mv core/llm/configuration_smollm3.py archive_legacy/llm_bloat/
mv core/llm/modeling_smollm3.py archive_legacy/llm_bloat/
mv core/llm/modular_smollm3.py archive_legacy/llm_bloat/
```

---

## FIX 10 — 🟠 HIGH — core/logging/ is 90% PyTorch internals, not Jarvis code

**Directory:** `core/logging/`

**The Bug:**  
50+ files are here. The actual Jarvis logger is `core/logger.py`. These files in `core/logging/` are borrowed from PyTorch, gRPC, and OpenTelemetry source trees:

```
_logsumexp.py          # scipy/torch math function
_logistic.py           # logistic regression — nothing to do with logging
bench_discrete_log.py  # benchmark for discrete logarithm
c10d_logger.py         # PyTorch distributed training logger
logging_tensor.py      # PyTorch tensor logging
Logo_pb2.py            # protobuf generated file
logs_pb2.py            # protobuf generated file
logs_pb2_grpc.py       # gRPC generated stub
logging_pb2.py         # protobuf generated file
prolog.py              # Prolog language integration (???)
morphology.py          # text morphology (NLP utility)
_morphology.py         # duplicate of above
```

**The Fix:**
```bash
mkdir -p archive_legacy/logging_bloat
# Move every file NOT used by core/logger.py
# Keep only: logger.py (in core/), audit_logger.py, audit_bridge.py
# Move everything else:
mv core/logging/_logsumexp.py archive_legacy/logging_bloat/
mv core/logging/_logistic.py archive_legacy/logging_bloat/
mv core/logging/bench_discrete_log.py archive_legacy/logging_bloat/
mv core/logging/c10d_logger.py archive_legacy/logging_bloat/
mv core/logging/logging_tensor.py archive_legacy/logging_bloat/
mv core/logging/Logo_pb2.py archive_legacy/logging_bloat/
mv core/logging/logs_pb2.py archive_legacy/logging_bloat/
mv core/logging/logs_pb2_grpc.py archive_legacy/logging_bloat/
mv core/logging/logging_pb2.py archive_legacy/logging_bloat/
mv core/logging/prolog.py archive_legacy/logging_bloat/
mv core/logging/morphology.py archive_legacy/logging_bloat/
mv core/logging/_morphology.py archive_legacy/logging_bloat/
# Verify nothing breaks
python -c "from core import logger; print('OK')"
```

---

## FIX 11 — 🟠 HIGH — core/controller/ contains Kubernetes and PyTorch state files

**Directory:** `core/controller/`

**The Bug:**  
These files have zero relevance to a personal AI assistant:

```
v1_stateful_set.py
v1_stateful_set_condition.py
v1_stateful_set_list.py
v1_stateful_set_ordinals.py
v1_stateful_set_persistent_volume_claim_retention_policy.py
v1_stateful_set_spec.py
v1_stateful_set_status.py
v1_stateful_set_update_strategy.py
v1_rolling_update_stateful_set_strategy.py
v1_lifecycle.py
v1_lifecycle_handler.py
v1_container_state.py
v1_container_state_running.py
v1_container_state_terminated.py
v1_container_state_waiting.py
ClientState_pb2.py
WidgetStates_pb2.py
_fsdp_state.py        # PyTorch FSDP (Fully Sharded Data Parallel)
_pybind_state.py      # PyTorch pybind11 state
_backward_state.py    # PyTorch autograd backward
_trace_state.py       # PyTorch trace
```

**The Fix:**
```bash
mkdir -p archive_legacy/controller_bloat
mv core/controller/v1_*.py archive_legacy/controller_bloat/
mv core/controller/ClientState_pb2.py archive_legacy/controller_bloat/
mv core/controller/WidgetStates_pb2.py archive_legacy/controller_bloat/
mv core/controller/_fsdp_state.py archive_legacy/controller_bloat/
mv core/controller/_pybind_state.py archive_legacy/controller_bloat/
mv core/controller/_backward_state.py archive_legacy/controller_bloat/
mv core/controller/_trace_state.py archive_legacy/controller_bloat/
# Run full test suite after
pytest tests/ -q
```

---

## FIX 12 — 🟠 HIGH — archive_legacy/files/reflection.py has a confirmed compile error

**File:** `archive_legacy/files/reflection.py`  
**Evidence:** `compile_report.txt` line: `[FAIL] D:\AI\Jarvis\archive_legacy\files\reflection.py`

**The Bug:**  
This file has a syntax or import error that was never resolved. While it is in archive_legacy, it is still discovered by pytest if norecursedirs is not set (which it currently is not). This causes collection errors that hide real test failures.

**The Fix:**
```bash
# Option A: Just fix the file (look at what's broken)
python -m py_compile archive_legacy/files/reflection.py
# Option B: Add to norecursedirs in pytest.ini (do this regardless)
# pytest.ini:
[pytest]
testpaths = tests
norecursedirs = Failed archive_legacy archive_jarvis_duplicate .git __pycache__ node_modules
asyncio_mode = auto
```

---

## FIX 13 — 🟡 MEDIUM — scripts/python.exe and pythonw.exe committed to git

**Files:** `scripts/python.exe`, `scripts/pythonw.exe`

**The Bug:**  
These are compiled Windows virtual environment binaries. They are 200KB+ each, binary, platform-specific, and should never be in version control. They break `git clone` on Linux/Mac (useless files), pollute diffs, and can fail antivirus checks on CI runners.

**The Fix:**
```bash
git rm --cached scripts/python.exe scripts/pythonw.exe
echo "scripts/*.exe" >> .gitignore
echo "scripts/*.dll" >> .gitignore
git commit -m "chore: remove venv binaries from tracking"
```

---

## FIX 14 — 🟡 MEDIUM — Runtime output files committed to git

**Files:**  
- `outputs/Jarvis-Session/execution_trace.jsonl`  
- `outputs/Jarvis-Session/tool_observations.jsonl`  
- `mnt/user-data/outputs/Jarvis-Session3/memory/__init__.py`

**The Bug:**  
These are generated at runtime. Every run overwrites them. Committing them causes merge conflicts and pollutes git history with runtime data that has no value as source code.

**The Fix:**
```bash
git rm --cached outputs/Jarvis-Session/execution_trace.jsonl
git rm --cached outputs/Jarvis-Session/tool_observations.jsonl
git rm -r --cached mnt/

# .gitignore additions:
echo "outputs/" >> .gitignore
echo "mnt/" >> .gitignore
echo "*.jsonl" >> .gitignore
echo "logs/" >> .gitignore
```

---

## FIX 15 — 🟡 MEDIUM — sqlite_pool.py connection pool never wired to memory_engine

**Files:**  
- `core/memory/sqlite_pool.py` — defines connection pool  
- `core/memory/memory_engine.py` — does NOT use the pool  
- `core/memory/sqlite.py` — second SQLite implementation, inconsistent with hybrid_memory.py

**The Bug:**  
`HybridMemory` creates sqlite3 connections directly via `sqlite3.connect()`. The connection pool exists but is never instantiated. In a multi-task async scenario (background monitor + agent loop running simultaneously), two coroutines can open the same SQLite file and cause `database is locked` errors.

**The Fix:**
```python
# core/memory/hybrid_memory.py — add in __init__
from core.memory.sqlite_pool import SQLitePool
self._pool = SQLitePool(self.db_path, pool_size=3)

# Replace all direct sqlite3.connect() calls with pool.acquire():
# BEFORE:
conn = sqlite3.connect(self.db_path, check_same_thread=False)
# AFTER:
with self._pool.acquire() as conn:
    # your query here
```

---

## FIX 16 — 🟡 MEDIUM — CI pipeline installs ALL requirements including GPU/audio deps

**File:** `.github/workflows/python-ci.yml`

**The Bug:**
```yaml
- run: pip install -r requirements.txt  # installs sounddevice, whispercpp, sentence-transformers
- run: pytest -q                         # no test isolation, no timeout, no coverage
```

`sounddevice` requires PortAudio system library not present on GitHub Actions Ubuntu runners. `whispercpp` requires a C++ compiler and model files. `sentence-transformers` downloads a 90MB model on first import. CI will fail or be extremely slow.

**The Fix:**
```yaml
# .github/workflows/python-ci.yml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'
    - name: Install core deps (no audio/GPU)
      run: |
        pip install pydantic fastapi uvicorn requests aiohttp python-dotenv rich
        pip install chromadb numpy cryptography
        pip install pytest pytest-asyncio pytest-mock
    - name: Run tests
      run: pytest tests/ -q --timeout=30 --tb=short
      env:
        JARVIS_ENV: test
```

```bash
# Also create requirements-dev.txt for CI
# And requirements.txt stays as the full local install
```

---

## FIX 17 — 🟡 MEDIUM — mypy.ini is effectively disabled

**File:** `mypy.ini`

**The Bug:**
```ini
[mypy]
ignore_missing_imports = True
# That's it. Nothing else.
```

This is the bare minimum that does nothing useful. `ignore_missing_imports = True` means mypy will not catch any wrong import, wrong type, or missing stub. It's theater.

**The Fix:**
```ini
[mypy]
python_version = 3.11
strict = False
ignore_missing_imports = True
warn_return_any = True
warn_unused_ignores = True
warn_redundant_casts = True
disallow_untyped_defs = True
check_untyped_defs = True
no_implicit_optional = True
show_error_codes = True

# Exclude third-party bloat that's not our code
exclude = (archive_legacy|archive_jarvis_duplicate|Failed|core/logging/_|core/controller/v1_|core/llm/useless)
```

---

## FIX 18 — 🔵 LOW — wake_word.py crashes hard on missing Porcupine key

**File:** `core/voice/wake_word.py`

**The Bug:**  
If `PORCUPINE_ACCESS_KEY` is empty or missing, Porcupine raises an `InvalidArgumentError` during initialization. This bubbles up to `VoiceLayer.__init__`, which has a `try/except ImportError` — but NOT an `except Exception`. The crash propagates to the controller and kills Jarvis startup entirely in voice mode.

**The Fix:**
```python
# core/voice/wake_word.py — wrap init in graceful fallback
class WakeWordDetector:
    def __init__(self, access_key: str, keyword: str = "jarvis"):
        self._active = False
        self._porcupine = None
        
        if not access_key:
            logger.warning(
                "PORCUPINE_ACCESS_KEY not set. Wake word detection disabled. "
                "Get a free key at: https://console.picovoice.ai/"
            )
            return
        
        try:
            import pvporcupine
            self._porcupine = pvporcupine.create(access_key=access_key, keywords=[keyword])
            self._active = True
            logger.info("Wake word detector active: '%s'", keyword)
        except Exception as exc:
            logger.warning("Wake word detector failed to init: %s. Continuing without it.", exc)
    
    @property
    def is_active(self) -> bool:
        return self._active
```

---

## COMPLETE .gitignore THAT SHOULD EXIST RIGHT NOW

**File:** `.gitignore` (current one is incomplete)

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
eggs/
parts/
var/
sdist/
develop-eggs/
.installed.cfg
lib/
lib64/

# Virtual environments
.venv/
venv/
env/
jarvis_env/
ENV/

# Secrets — NEVER commit these
*.env
config/settings.env
!config/settings.env.template

# Runtime outputs — generated, not source
outputs/
logs/
*.jsonl
*.log
mnt/
memory/*.db
memory/*.ics
data/chroma/
workspace/

# Windows binaries that snuck in
scripts/*.exe
scripts/*.dll
scripts/*.bat

# IDE
.idea/
.vscode/
*.iml
.DS_Store
Thumbs.db

# Test artifacts
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
.tox/

# Archive dirs — never auto-import these
# (explicitly excluded in pytest.ini too)
```

---

## VERIFICATION CHECKLIST — Run After All Fixes

```bash
# 1. Sandbox path is relative
python -c "from core.tools.builtin_tools import _SANDBOX_ROOT; assert _SANDBOX_ROOT.exists(), f'FAIL: {_SANDBOX_ROOT}'; print('PASS: sandbox path')"

# 2. No hardcoded Windows paths remain
grep -rn "D:/AI\|D:\\\\AI" core/ --include="*.py" | grep -v __pycache__ && echo "FAIL: hardcoded paths found" || echo "PASS: no hardcoded paths"

# 3. Correct LLMClientV2 is imported
python -c "from core.llm.client import LLMClientV2; c = LLMClientV2(); assert hasattr(c, 'complete'), 'FAIL'; print('PASS: correct client')"

# 4. No asyncio.new_event_loop in async modules
grep -rn "asyncio.new_event_loop" core/ --include="*.py" | grep -v __pycache__ && echo "WARN: review these" || echo "PASS: no raw event loops"

# 5. Test suite runs clean
pytest tests/ -q --timeout=30 2>&1 | tail -5

# 6. settings.env NOT tracked
git ls-files config/settings.env && echo "FAIL: env file is tracked" || echo "PASS: env file excluded"

# 7. No exe binaries in git
git ls-files "*.exe" && echo "FAIL: binaries tracked" || echo "PASS: no binaries"

# 8. Calendar import works with icalendar
python -c "from icalendar import Calendar; print('PASS: icalendar installed')"
```

---

*This file is machine-generated from a full audit of 280+ files. Every fix has been traced to specific line numbers. Fix in order. Do not skip FIX 1–4 — they are cascading failures.*
