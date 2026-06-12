# Medium Priority Issues

### AUDIT-003 - Security vulnerabilities
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `_LONG_SECRET` regex fails to redact base64-encoded secrets or complex tokens containing non-alphanumeric characters (such as `+`, `/`, `=`, `-`, `_`).
**Root Cause:** The character class `[a-zA-Z0-9]` strictly restricts matches to alphanumeric characters, breaking the match continuity when special token characters appear.
**Impact:** Secure tokens that utilize standard base64 or URL-safe encoding will bypass redaction entirely if they lack a continuous 32-character alphanumeric chunk.
**Fix:** 

### AUDIT-004 - Logic errors
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `scrub_secrets` function incorrectly silences valid falsy inputs (such as the integer `0`, `0.0`, or boolean `False`), blindly converting them to empty strings.
**Root Cause:** The expression `text or ""` evaluates to an empty string for all falsy values, rather than just handling `None`.
**Impact:** Audit logs will unexpectedly drop valid data, leading to incomplete or inaccurate audit trails when logging payloads that contain zero values or boolean flags.
**Fix:** 

### JARVIS-CONFIG-002 - Configuration problems
**Files:** d:\AI\Jarvis\config\settings.env, d:\AI\Jarvis\config\settings.env.template
**Description:** The active environment file (`settings.env`) is severely out of sync with its template (`settings.env.template`), missing multiple critical configuration blocks.
**Root Cause:** `settings.env` was likely not updated when new integrations (Home Assistant, GitHub, Quick local model routing) were added to the `.template` file.
**Impact:** Attempting to use Home Assistant or GitHub features will fail because the required credentials and URLs are absent from the active environment variables, leading to broken integrations.
**Fix:** 

### JARVIS-CONFIG-004 - Configuration problems
**Files:** d:\AI\Jarvis\config\jarvis.ini
**Description:** Duplicate and redundant model configurations between the `[ollama]` and `[models]` sections.
**Root Cause:** The models `vision_model = llava` and `plan(ner)_model = deepseek-r1:8b` are configured redundantly in both the `[ollama]` block and the general `[models]` block.
**Impact:** Changing a model in one section may not take effect if the system reads from the other section, leading to the wrong models being executed for tasks and causing unexpected behaviors.
**Fix:** 

### JARVIS-CORE-002 - Async issues
**Files:** `d:\AI\Jarvis\core\controller_v2.py`, `d:\AI\Jarvis\core\profile.py`, `d:\AI\Jarvis\core\controller\goal_runner.py`, `d:\AI\Jarvis\core\automation\live_automation.py`
**Description:** Synchronous file I/O operations block the main asyncio event loop across multiple components. Specifically, updating profiles, persisting goal states, and saving background automation states write directly to disk without offloading the work to threads.
**Root Cause:** Calling synchronous standard library I/O (like `write_text`, `open`, `os.replace`, and `json.dump`) inside async functions or event loop scheduled callbacks instead of offloading them.
**Impact:** Spikes in disk I/O will stall the event loop, causing the system to appear unresponsive, delaying automated tasks, and stuttering active integrations like voice synthesis.
**Fix:** 

### JARVIS-CORE-003 - Logic errors
**Files:** `d:\AI\Jarvis\core\controller\intent_handlers.py`, `d:\AI\Jarvis\core\controller_v2.py`
**Description:** The intent handler for "automation" commands expects the `live_automation` engine to be mounted directly on the context (`ctx.live_automation`). However, `controller_v2.py` wraps it in an `automation_manager` subsystem, leaving the handler trying to extract `None`.
**Root Cause:** Structural desynchronization after a facade/subsystem refactor. The intent route checking logic was not updated to reflect the new nested attribute path.
**Impact:** The intent router completely bypasses any commands starting with "automation scan", "automation status", or "rag search " because the condition evaluates to `None`, making manual RAG querying unavailable to the user.
**Fix:** 

### DASHBOARD-003 - Async issues
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** Synchronous component calls (`_load_ai_os_overview`, `manager.create()`, `manager.complete()`, `_load_active_goals()`) are executed directly inside `async def` route handlers, potentially blocking the event loop with disk/database I/O.
**Root Cause:** Route handlers like `ai_os_page`, `api_ai_os`, `goals_add`, and `goals_complete` are declared as `async def` but call synchronous functions that perform disk reads (loading YAML blueprints) or database queries without offloading them.
**Impact:** Momentary unresponsiveness of the web dashboard and other async tasks when users load the AI OS page or manage goals.
**Fix:** 

### DesignCHnage-001 - Error handling gaps
**Files:** d:\AI\Jarvis\DesignCHnage\investigate_domain.py
**Description:** Unhandled `FileNotFoundError` in the `update_ledger()` function.
**Root Cause:** The `update_ledger` function assumes `COVERAGE_LEDGER_PATH` exists without checking, whereas the `parse_ledger` function explicitly handles its absence gracefully. If the file is missing, the script will crash inside `update_ledger()`.
**Impact:** Script fails with an exception if the ledger file is deleted or renamed, halting execution.
**Fix:** 

### DOCS-001 - Documentation mismatches
**Files:** d:\AI\Jarvis\docs\design_doc.md
**Description:** The design document embeds images using hardcoded, absolute local file URIs that point to a temporary AI workspace, which will be broken for any other user or system.
**Root Cause:** Images were referenced directly from an LLM chat brain directory during document creation, rather than being saved into the repository as persistent relative assets.
**Impact:** Users reading the documentation will not be able to view the embedded diagrams, resulting in a loss of critical visual context.
**Fix:** 

### FINAL-002 - Architectural inconsistencies
**Files:** d:\AI\Jarvis\Final\01_Architecture_Map.md, d:\AI\Jarvis\Final\Architecture_Map.md, d:\AI\Jarvis\Final\02_Execution_Graph.md, d:\AI\Jarvis\Final\Execution_Graph.md
**Description:** Duplicate files exist for both the Architecture Map and the Execution Graph, and they contain contradictory implementation details regarding system coupling and initialization.
**Root Cause:** Multiple versions of system diagrams were generated or retained during documentation iterations without cleaning up redundant or obsolete files.
**Impact:** Creates architectural ambiguity for engineers regarding whether the system uses dynamic string-based class loading or direct explicit instantiation.
**Fix:** 

### INTEGRATIONS-004 - Dependency issues
**Files:** `integrations/clients/calendar.py`
**Description:** The `calendar` integration returns `True` unconditionally in its `is_available()` method, but its `_list_events` utility relies heavily on the third-party libraries `icalendar` and `dateutil`. If these libraries are missing, the integration registers successfully but throws runtime exceptions.
**Root Cause:** `is_available()` ignores the dynamic imports made within the class methods.
**Impact:** Integration is actively exposed to the planner but crashes abruptly with an `ImportError` upon execution.
**Fix:** 

### INTEGRATIONS-005 - Architectural inconsistencies
**Files:** `integrations/registry.py`
**Description:** The `IntegrationRegistry.get_tools()` method iterates over all integrated instances and returns their tool schemas without checking the active `self._tool_owner` mapping. If an integration overrides a tool name belonging to another integration, both the overridden and overriding tool schemas are exported.
**Root Cause:** The iteration over `self._integrations.values()` blindly appends all schemas without validating tool ownership.
**Impact:** The system outputs duplicate/stale tool schemas to the planner agent, causing LLM confusion, context payload bloat, and unpredictable routing behavior.
**Fix:** 

### LOGS-003 - File handling issues
**Files:** d:\AI\Jarvis\logs\audit.jsonl.corrupted
**Description:** The audit log file is marked as corrupted, indicating a failure to properly write or close the JSONL audit file.
**Root Cause:** Likely an improper application shutdown, an unexpected crash, or a race condition during file writing that caused the JSONL structure to become malformed or truncated.
**Impact:** Loss of critical audit trails or failures in the dashboard/log viewer when attempting to parse the audit logs.
**Fix:** 

### REQUIREMENTS-001 - Dependency issues
**Files:** d:\AI\Jarvis\requirements\full.txt
**Description:** Use of the deprecated and unmaintained `fpdf` library.
**Root Cause:** The original `fpdf` package has been abandoned for years and does not support modern Python environments. The officially maintained successor is `fpdf2`.
**Impact:** Lack of security patches, missing modern features (like proper UTF-8 support), and potential compatibility failures with newer Python versions.
**Fix:** 

### 4 - Data flow
**Files:** core/controller_v2.py, core/controller/goal_runner.py
**Description:** During a graceful shutdown, the system cancels the background `_goal_check_task` but fails to explicitly flush and persist in-memory goal mutations to disk.
**Root Cause:** Missing persistence hook during the controller's shutdown sequence.
**Impact:** Any active goals that were dynamically modified, updated, or added shortly before shutdown could be permanently lost, causing state regression on the next boot.
**Fix:** 

### 5 - Build system consistency
**Files:** requirements.lock, jarvis.spec, requirements/desktop.txt, requirements/voice.txt
**Description:** The `requirements.lock` file only resolves dependencies for the base runtime, omitting optional capabilities. Conversely, the PyInstaller build script (`jarvis.spec`) hardcodes these optional modules into its `hiddenimports` list.
**Root Cause:** `requirements.lock` was generated against `requirements/base.txt` rather than `requirements/full.txt`, while PyInstaller expects a fully-featured environment to compile.
**Impact:** Attempting to build the binary using the strict lockfile will cause PyInstaller to crash due to missing modules. Additionally, users installing purely from the lockfile will unknowingly lack voice and desktop automation features.
**Fix:** 

