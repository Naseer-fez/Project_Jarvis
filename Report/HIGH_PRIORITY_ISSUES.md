# High Priority Issues

### AUDIT-002 - Security vulnerabilities
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `_ASSIGNMENT_PATTERNS` regex strictly expects an equals sign (`=`) separator, failing to redact secrets embedded in JSON payloads or Python dictionaries where colons (`:`) are used.
**Root Cause:** The assignment regex `\s*=\s*` hardcodes the equals sign as the only valid key-value separator.
**Impact:** Applications logging JSON requests, responses, or dictionary objects will leak secret values in plaintext.
**Fix:** 

### JARVIS-CONFIG-001 - Missing implementations
**Files:** d:\AI\Jarvis\config\jarvis.ini
**Description:** The configuration file references a blueprint file that does not exist in the config directory.
**Root Cause:** The `[ai_os]` section sets `blueprint_file = config/ai_os.json`, but `ai_os.json` is completely missing from the `config` folder.
**Impact:** Will cause a FileNotFoundError or application crash when the AI OS module attempts to read its blueprint file upon initialization.
**Fix:** 

### JARVIS-CONFIG-005 - Security vulnerabilities
**Files:** d:\AI\Jarvis\config\jarvis.ini
**Description:** Conflicting risk categories. The `forbidden_actions`, `blocked_actions`, and `critical_actions` configuration keys are identically assigned the exact same list of dangerous actions.
**Root Cause:** A copy-paste error during the creation of the `[risk]` configuration section.
**Impact:** Critical security risk. If the permissions logic checks `critical_actions` before `forbidden_actions`, an attacker or runaway agent could execute a forbidden action (like `format_disk` or `execute_shell`) simply by passing/bypassing the "critical" user confirmation prompt.
**Fix:** 

### JARVIS-CORE-001 - Race conditions
**Files:** `d:\AI\Jarvis\core\controller_v2.py`
**Description:** The `current_classification` attribute is stored directly on the `JarvisControllerV2` singleton instance inside the async `process()` method without locking. Because `process()` yields execution to an intent router before the classification is consumed by `_dispatch_llm`, concurrent user requests will overwrite each other's classification state.
**Root Cause:** Storing request-specific execution state (`current_classification`) on the instance rather than passing it explicitly down the function call chain as a local variable.
**Impact:** Under concurrent load, the controller might assign complex tasks to simple chat pathways (or vice-versa), bypassing the planner or failing to allocate sufficient context, leading to request failures.
**Fix:** 

### DASHBOARD-001 - Security vulnerabilities
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** The `/gui-audit` route is mounted using FastAPI's `StaticFiles` without any authentication checks, exposing sensitive screenshots of the user's PC to anyone who can access the dashboard's network port.
**Root Cause:** `app.mount("/gui-audit", StaticFiles(...))` mounts the directory statically at the application root, bypassing the authentication enforcement (`_is_authorized`) applied to the API and page routes.
**Impact:** Unauthorized users on the local network (or internet, if the port is exposed) can access and view screenshots of the user's desktop, leading to a severe privacy and security breach.
**Fix:** 

### DASHBOARD-002 - Async issues
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** The `/api/convert` endpoint performs blocking file operations and synchronous conversion directly on the main event loop thread.
**Root Cause:** The `api_convert` function is defined with `async def`, causing FastAPI to schedule it on the main asyncio event loop. Inside, it uses synchronous blocking operations like `shutil.copyfileobj(file.file, tmp_src)` and `perform_conversion(...)` without delegating them to a thread pool.
**Impact:** The entire FastAPI application (including health checks, WebSockets, and other user requests) will be unresponsive while a file is being uploaded, copied, or converted.
**Fix:** 

### DASHBOARD-004 - Data validation issues
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** The `target_format` field in the `/api/convert` endpoint is not validated or sanitized for special characters before being passed to the converter.
**Root Cause:** The endpoint accepts any string from the `Form(...)` input, only applying `.strip().lower()`. It does not restrict the string to alphanumeric characters, allowing potential path traversal payloads or command injection payloads to be passed to `perform_conversion`.
**Impact:** An authenticated attacker could supply a crafted `target_format` (e.g., `../evil.sh`) to overwrite arbitrary files on the system, or potentially execute arbitrary code if the underlying converter invokes a shell command.
**Fix:** 

### DesignCHnage-003 - Logic errors
**Files:** d:\AI\Jarvis\DesignCHnage\orchestrate_recovery.py
**Description:** The script unconditionally overwrites existing detailed architecture, validation, and rebuild documentation files with short hardcoded stubs.
**Root Cause:** In `orchestrate_recovery.py`, the `open()` function is used with mode `"w"` unconditionally for all markdown files. Since the actual documentation files (e.g. `Architecture_Overview.md` which is ~1800 bytes) contain detailed content, running this script truncates and overwrites them with the short stub strings defined in the script's dictionaries.
**Impact:** Severe data loss; running the script inadvertently destroys rich, previously generated documentation artifacts across four directories.
**Fix:** 

### FINAL-001 - Documentation mismatches
**Files:** d:\AI\Jarvis\Final\Phase_2_3_4_Report.md, d:\AI\Jarvis\Final\08_Phase_2_Root_Cause_Analysis.md, d:\AI\Jarvis\Final\09_Phase_3_Recovery_Strategy.md
**Description:** There is a major documentation mismatch regarding the root causes and recovery strategies for the system. The standalone phase reports (`08_` and `09_`) focus on asynchronous lifecycles, dynamic class loading fragility, and event loop blocking as the core issues. In contrast, the consolidated `Phase_2_3_4_Report.md` completely contradicts this, identifying duplicate system architectures (`core.agentic` vs `core.autonomy`) and monolithic controllers as the root causes to solve.
**Root Cause:** The markdown reports were likely generated by parallel, uncoordinated documentation efforts or differing analysis subagents that assessed different aspects of the codebase without consolidating their conflicting recovery strategies.
**Impact:** Agents or developers relying on these blueprints for Phase 5 implementation will face conflicting instructions, leading to split architectural focus and potentially breaking the system by executing the wrong recovery plan.
**Fix:** 

### INTEGRATIONS-001 - Security vulnerabilities
**Files:** `integrations/clients/computer_control.py`
**Description:** The `take_screenshot` tool accepts a user-provided `path` and passes it directly to `os.path.abspath` without verifying that the resolved path is within a safe, intended directory. This permits an arbitrary file write (path traversal vulnerability) anywhere on the system.
**Root Cause:** Lack of directory boundary checking before writing files to the disk.
**Impact:** Arbitrary file overwrite, potentially replacing critical system files or injecting malicious files.
**Fix:** 

### INTEGRATIONS-002 - Race conditions
**Files:** `integrations/clients/calendar.py`
**Description:** The `_add_event` method executes within an asynchronous thread pool but performs read-modify-write operations on `CALENDAR_PATH` without locks, exposing it to race conditions. Additionally, it uses naive string replacement `.replace("END:VCALENDAR", block + "END:VCALENDAR")` which will aggressively replace all occurrences, including any `END:VCALENDAR` strings maliciously or accidentally included in an event title, severely corrupting the file.
**Root Cause:** Missing file locking for concurrent thread pool operations and usage of fragile string manipulation instead of robust ICS object parsing.
**Impact:** Complete corruption of the `.ics` calendar file or loss of events when concurrent calls are made.
**Fix:** 

### INTEGRATIONS-003 - Logic errors
**Files:** `integrations/clients/google_calendar.py`
**Description:** When parsing the start and end times for creating an event, `_to_rfc3339` unconditionally attaches UTC `timezone.utc` to any naive datetimes. When combined with the intended `timeZone` field in the API payload, Google Calendar interprets the absolute time as UTC and incorrectly shifts it into the provided timezone.
**Root Cause:** `dt.replace(tzinfo=timezone.utc)` blindly overrides the user's intended local time when no offset is specified.
**Impact:** Events get scheduled at incorrect offset hours resulting in entirely wrong meeting times.
**Fix:** 

### LOGS-001 - Dependency issues
**Files:** d:\AI\Jarvis\logs\app.log
**Description:** The application logs indicate missing dependencies for critical voice components: `faster-whisper` (STT), `piper-tts` (TTS), and `openwakeword` (Wake Word). The application is forced to fallback to dummy or CLI implementations.
**Root Cause:** The Python environment running Jarvis is missing required packages and model files (e.g., `en_US-lessac-medium.json`, `alexa_v0.1.onnx`).
**Impact:** Voice interaction is completely disabled or using unusable mock fallbacks, breaking voice-related features.
**Fix:** 

### LOGS-002 - Integration failures
**Files:** d:\AI\Jarvis\logs\app.log, d:\AI\Jarvis\logs\audit.jsonl.bak
**Description:** The application repeatedly fails to connect to Ollama and cloud LLM providers, leading to timeout errors and planner degradation.
**Root Cause:** The Ollama service is either not running locally, or the network configuration is preventing access. Additionally, cloud fallbacks are left unconfigured.
**Impact:** The core agent loop and planning capabilities cannot function properly without an LLM, rendering the AI assistant mostly inoperable.
**Fix:** 

### MEMORY-001 - Data validation issues
**Files:** d:\AI\Jarvis\memory\user_profile.json
**Description:** The 'name' field in user_profile.json is set to "hacked", which strongly indicates a potential data validation issue, unauthorized modification, or state corruption risk.
**Root Cause:** Probable lack of strict input validation or sanitization on user profile updates, allowing malicious or unexpected strings to be written into the profile's 'name' field. Alternatively, it could be a deliberate placeholder, but it warrants immediate investigation for unauthorized access.
**Impact:** If this value is used in templates, logs, or downstream systems without escaping, it could lead to further injection attacks. It also represents a compromised or unexpected system state.
**Fix:** 

### JARVIS-TESTS-001 - Broken tests
**Files:** d:\AI\Jarvis\tests\unit\test_agent_loop.py
**Description:** The `test_agent_loop_user_interrupt` test passes a synchronous lambda function (`lambda prompt: False`) as the `confirm_callback` argument to `engine.run()`. The `AgentLoopEngine` expects this callback to be an asynchronous function, as it is awaited internally.
**Root Cause:** A synchronous lambda was provided where an asynchronous callable is required. This results in a runtime `TypeError` (`object bool can't be used in 'await' expression`) when the callback is awaited.
**Impact:** The test will consistently fail at runtime, blocking CI/CD pipelines and preventing the validation of user interrupt logic.
**Fix:** 

### 1 - Dependencies between folders
**Files:** core/runtime/dashboard_runtime.py, dashboard/server.py
**Description:** There is a circular dependency between the core subsystem and the dashboard. The `core` module explicitly imports `dashboard.server` to bootstrap the web UI, while `dashboard.server` heavily imports from multiple `core.*` namespaces (such as `core.security.auth`, `core.ai_os`, and `core.plugins`).
**Root Cause:** Lack of architectural boundary enforcement. `DashboardRuntime` is nested inside `core` but is tightly coupled with the `dashboard` module, creating a dependency cycle.
**Impact:** This breaks module isolation, impedes refactoring, and can cause unpredictable `ImportError`s during initialization due to circular loading.
**Fix:** 

### 3 - Shutdown sequence
**Files:** core/runtime/dashboard_runtime.py, core/runtime/entrypoint.py
**Description:** The shutdown sequence blocks the main asyncio event loop synchronously for up to 5 seconds when stopping the dashboard server.
**Root Cause:** A blocking `Thread.join()` operation is executed directly inside the main `async def` thread without offloading it.
**Impact:** Freezes the entire asyncio event loop during shutdown. This prevents pending asynchronous cleanup tasks (like memory flush or state machine termination) from executing, potentially leading to hard crashes or deadlocks.
**Fix:** 

