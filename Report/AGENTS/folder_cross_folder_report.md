ISSUE ID: 1
SEVERITY: High
CATEGORY: Dependencies between folders
FILES: core/runtime/dashboard_runtime.py, dashboard/server.py
DESCRIPTION: There is a circular dependency between the core subsystem and the dashboard. The `core` module explicitly imports `dashboard.server` to bootstrap the web UI, while `dashboard.server` heavily imports from multiple `core.*` namespaces (such as `core.security.auth`, `core.ai_os`, and `core.plugins`).
ROOT CAUSE: Lack of architectural boundary enforcement. `DashboardRuntime` is nested inside `core` but is tightly coupled with the `dashboard` module, creating a dependency cycle.
EVIDENCE: `core/runtime/dashboard_runtime.py` contains `from dashboard.server import app`. Meanwhile, `dashboard/server.py` contains multiple core imports like `from core.security.auth import AuthManager`.
POTENTIAL IMPACT: This breaks module isolation, impedes refactoring, and can cause unpredictable `ImportError`s during initialization due to circular loading.
RECOMMENDED FIX: Decouple the startup sequence by injecting the FastAPI app instance into the runtime from the main entrypoint, or move `DashboardRuntime` out of `core` entirely.

ISSUE ID: 2
SEVERITY: Critical
CATEGORY: Service interactions
FILES: core/runtime/bootstrap.py, core/registry/registry.py, integrations/registry.py, core/controller/services.py
DESCRIPTION: Integrations (e.g., GitHub, Notion, Calendar) are successfully loaded into an isolated `IntegrationRegistry` at startup. However, the core execution engine and task planner exclusively query `CapabilityRegistry` (`tool_router`). These two registries are never bridged, rendering the integrations invisible to the LLM.
ROOT CAUSE: Disconnected service registries. `bootstrap.py` initializes the integration registry and attaches it to the controller, but fails to inject or register these tools with the `tool_router` used by the execution loop.
EVIDENCE: `bootstrap.py` executes `integration_registry.register_safety_rules(...)`. However, `services.py` injects `CapabilityRegistry` as the sole `tool_router`, which only processes `builtin_tools` and dynamic plugins, completely ignoring `IntegrationRegistry`.
POTENTIAL IMPACT: All integration modules residing in the `integrations/clients/` directory are orphaned. The agent will never execute or discover external integration tools.
RECOMMENDED FIX: Modify `bootstrap.py` or `services.py` to iterate over `IntegrationRegistry.get_tools()` and dynamically map them into `CapabilityRegistry` as `FunctionCapability` wrappers, or consolidate the two registries into a single system.

ISSUE ID: 3
SEVERITY: High
CATEGORY: Shutdown sequence
FILES: core/runtime/dashboard_runtime.py, core/runtime/entrypoint.py
DESCRIPTION: The shutdown sequence blocks the main asyncio event loop synchronously for up to 5 seconds when stopping the dashboard server.
ROOT CAUSE: A blocking `Thread.join()` operation is executed directly inside the main `async def` thread without offloading it.
EVIDENCE: `DashboardRuntime.stop()` explicitly calls `self._thread.join(timeout=timeout)`. `core/runtime/entrypoint.py` calls this method directly within the `finally` block of the `async_run` coroutine.
POTENTIAL IMPACT: Freezes the entire asyncio event loop during shutdown. This prevents pending asynchronous cleanup tasks (like memory flush or state machine termination) from executing, potentially leading to hard crashes or deadlocks.
RECOMMENDED FIX: Offload the blocking call by executing `await asyncio.to_thread(dashboard.stop, timeout)` in `entrypoint.py`, or refactor `DashboardRuntime` to use an async-native teardown.

ISSUE ID: 4
SEVERITY: Medium
CATEGORY: Data flow
FILES: core/controller_v2.py, core/controller/goal_runner.py
DESCRIPTION: During a graceful shutdown, the system cancels the background `_goal_check_task` but fails to explicitly flush and persist in-memory goal mutations to disk.
ROOT CAUSE: Missing persistence hook during the controller's shutdown sequence.
EVIDENCE: `JarvisControllerV2.shutdown()` successfully cancels and awaits `self._goal_check_task`, but it does not invoke `self.goal_runner.persist_goal_state()`.
POTENTIAL IMPACT: Any active goals that were dynamically modified, updated, or added shortly before shutdown could be permanently lost, causing state regression on the next boot.
RECOMMENDED FIX: Explicitly invoke `self.goal_runner.persist_goal_state()` inside `JarvisControllerV2.shutdown()` prior to completing the teardown.

ISSUE ID: 5
SEVERITY: Medium
CATEGORY: Build system consistency
FILES: requirements.lock, jarvis.spec, requirements/desktop.txt, requirements/voice.txt
DESCRIPTION: The `requirements.lock` file only resolves dependencies for the base runtime, omitting optional capabilities. Conversely, the PyInstaller build script (`jarvis.spec`) hardcodes these optional modules into its `hiddenimports` list.
ROOT CAUSE: `requirements.lock` was generated against `requirements/base.txt` rather than `requirements/full.txt`, while PyInstaller expects a fully-featured environment to compile.
EVIDENCE: `requirements.lock` is missing packages such as `pyautogui`, `speechrecognition`, and `opencv-python`. Meanwhile, `jarvis.spec` enforces `hiddenimports=['pyautogui', 'speech_recognition', 'cv2', ...]`.
POTENTIAL IMPACT: Attempting to build the binary using the strict lockfile will cause PyInstaller to crash due to missing modules. Additionally, users installing purely from the lockfile will unknowingly lack voice and desktop automation features.
RECOMMENDED FIX: Regenerate `requirements.lock` using `requirements/full.txt` to encompass all dependencies, or update `jarvis.spec` to dynamically detect installed modules rather than hardcoding them.

ISSUE ID: 6
SEVERITY: Low
CATEGORY: Dependencies between folders
FILES: audit/audit_logger.py, core/logging/logger.py
DESCRIPTION: The `audit` directory is orphaned. It contains secret scrubbing logic (`audit_logger.py`) that is disconnected from the active runtime. The actual logging system implements its own redundant data redaction.
ROOT CAUSE: Architectural divergence where logging was rebuilt and consolidated into `core.logging`, abandoning the top-level `audit` module without deprecation.
EVIDENCE: `grep` analysis confirms zero active imports of `audit.audit_logger` in the execution paths. `core/logging/logger.py` independently implements `redact_sensitive_data` and an `AuditLog` class.
POTENTIAL IMPACT: Accumulation of dead code, confusion for maintainers, and security risks if engineers attempt to patch the orphaned `audit` module instead of the active `core` module.
RECOMMENDED FIX: Delete the `audit` directory and its contents, or refactor `core.logging.logger` to import and utilize the existing scrubbing logic from `audit.audit_logger`.
