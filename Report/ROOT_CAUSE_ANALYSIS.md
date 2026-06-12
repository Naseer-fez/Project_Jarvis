# Root Cause Analysis

### AUDIT-001 (Critical)
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `_ASSIGNMENT_PATTERNS` regex fails to match common secret variables prefixed with other words or underscores (e.g., `db_password`, `user_token`) due to the `\b` word boundary anchor.

### AUDIT-002 (High)
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `_ASSIGNMENT_PATTERNS` regex strictly expects an equals sign (`=`) separator, failing to redact secrets embedded in JSON payloads or Python dictionaries where colons (`:`) are used.

### AUDIT-003 (Medium)
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `_LONG_SECRET` regex fails to redact base64-encoded secrets or complex tokens containing non-alphanumeric characters (such as `+`, `/`, `=`, `-`, `_`).

### AUDIT-004 (Medium)
**Files:** d:\AI\Jarvis\audit\audit_logger.py
**Description:** The `scrub_secrets` function incorrectly silences valid falsy inputs (such as the integer `0`, `0.0`, or boolean `False`), blindly converting them to empty strings.

### JARVIS-CONFIG-005 (High)
**Files:** d:\AI\Jarvis\config\jarvis.ini
**Description:** Conflicting risk categories. The `forbidden_actions`, `blocked_actions`, and `critical_actions` configuration keys are identically assigned the exact same list of dangerous actions.

### JARVIS-CORE-001 (High)
**Files:** `d:\AI\Jarvis\core\controller_v2.py`
**Description:** The `current_classification` attribute is stored directly on the `JarvisControllerV2` singleton instance inside the async `process()` method without locking. Because `process()` yields execution to an intent router before the classification is consumed by `_dispatch_llm`, concurrent user requests will overwrite each other's classification state.

### JARVIS-CORE-002 (Medium)
**Files:** `d:\AI\Jarvis\core\controller_v2.py`, `d:\AI\Jarvis\core\profile.py`, `d:\AI\Jarvis\core\controller\goal_runner.py`, `d:\AI\Jarvis\core\automation\live_automation.py`
**Description:** Synchronous file I/O operations block the main asyncio event loop across multiple components. Specifically, updating profiles, persisting goal states, and saving background automation states write directly to disk without offloading the work to threads.

### JARVIS-CORE-003 (Medium)
**Files:** `d:\AI\Jarvis\core\controller\intent_handlers.py`, `d:\AI\Jarvis\core\controller_v2.py`
**Description:** The intent handler for "automation" commands expects the `live_automation` engine to be mounted directly on the context (`ctx.live_automation`). However, `controller_v2.py` wraps it in an `automation_manager` subsystem, leaving the handler trying to extract `None`.

### JARVIS-CORE-004 (Low)
**Files:** `d:\AI\Jarvis\core\controller_v2.py`
**Description:** The CLI execution loop does not wrap the invocation of `await self.process(text)` in a `try/except` block.

### DASHBOARD-001 (High)
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** The `/gui-audit` route is mounted using FastAPI's `StaticFiles` without any authentication checks, exposing sensitive screenshots of the user's PC to anyone who can access the dashboard's network port.

### DASHBOARD-002 (High)
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** The `/api/convert` endpoint performs blocking file operations and synchronous conversion directly on the main event loop thread.

### DASHBOARD-003 (Medium)
**Files:** d:\AI\Jarvis\dashboard\server.py
**Description:** Synchronous component calls (`_load_ai_os_overview`, `manager.create()`, `manager.complete()`, `_load_active_goals()`) are executed directly inside `async def` route handlers, potentially blocking the event loop with disk/database I/O.

### DesignCHnage-001 (Medium)
**Files:** d:\AI\Jarvis\DesignCHnage\investigate_domain.py
**Description:** Unhandled `FileNotFoundError` in the `update_ledger()` function.

### DesignCHnage-003 (High)
**Files:** d:\AI\Jarvis\DesignCHnage\orchestrate_recovery.py
**Description:** The script unconditionally overwrites existing detailed architecture, validation, and rebuild documentation files with short hardcoded stubs.

### DesignCHnage-006 (Low)
**Files:** d:\AI\Jarvis\DesignCHnage\investigate_domain.py
**Description:** Incorrect counting of top-level functions in evidence reports.

### INTEGRATIONS-001 (High)
**Files:** `integrations/clients/computer_control.py`
**Description:** The `take_screenshot` tool accepts a user-provided `path` and passes it directly to `os.path.abspath` without verifying that the resolved path is within a safe, intended directory. This permits an arbitrary file write (path traversal vulnerability) anywhere on the system.

### INTEGRATIONS-002 (High)
**Files:** `integrations/clients/calendar.py`
**Description:** The `_add_event` method executes within an asynchronous thread pool but performs read-modify-write operations on `CALENDAR_PATH` without locks, exposing it to race conditions. Additionally, it uses naive string replacement `.replace("END:VCALENDAR", block + "END:VCALENDAR")` which will aggressively replace all occurrences, including any `END:VCALENDAR` strings maliciously or accidentally included in an event title, severely corrupting the file.

### INTEGRATIONS-003 (High)
**Files:** `integrations/clients/google_calendar.py`
**Description:** When parsing the start and end times for creating an event, `_to_rfc3339` unconditionally attaches UTC `timezone.utc` to any naive datetimes. When combined with the intended `timeZone` field in the API payload, Google Calendar interprets the absolute time as UTC and incorrectly shifts it into the provided timezone.

### INTEGRATIONS-006 (Low)
**Files:** `integrations/clients/gmail.py`
**Description:** In `_get_message_meta()`, if the HTTP request to fetch individual message metadata fails (e.g., due to a rate-limiting 429 or 404), the method silently parses the error payload and returns an empty dictionary.

