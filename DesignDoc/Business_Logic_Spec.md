# Business Logic Specification

This document maps all core business logic extracted from the `Jarvis` source code. It covers routing, validation, permissions, calculations, state transitions, and workflow executions.

## 1. Automation Routing Rules
**Description**: Defines how files dropped into specific folders are routed and processed by the live automation system.
**Trigger**: File ingestion detected in monitored directories.
**Conditions**:
- **Commands**: If file is in `commands_dir` and matches `command_extensions` → Route to `commands` kind. Sets `move_to_failed=True`, `mark_seen=False`, failure_label="Command ingestion".
- **RAG Data**: If file is in `rag_dir` → Route to `rag` kind. Sets `move_after=True`, `move_to_failed=True`, mark_seen=False.
- **Screenshots**: If `watch_screenshots=True` and file matches `image_extensions` in `screenshots_dir` → Route to `screenshots` kind. Sets `mark_seen=True`, `move_after=False`.
- **Recordings**: If `watch_recordings=True` and file matches `video_extensions` in `recordings_dir` → Route to `recordings` kind. Sets `mark_seen=True`, `move_after=False`.
**Outputs**: Constructs a `tuple` of `ScanRoute` configurations that the scanning pipeline consumes.
**Dependencies**: Directory paths configurations, feature flags (`watch_screenshots`, `watch_recordings`) (`core/automation/scan_rules.py`).

## 2. Request Complexity & Routing
**Description**: Calculates the complexity of a user prompt to determine if it needs deep reasoning, agentic planning, or a simple reflex reply.
**Trigger**: Every user query string processed by the system.
**Conditions**:
- **Base Classification:**
  - *Reflex*: Matches keywords (`weather`, `time`, `date`, `status`) or starts with "what is the". Base Complexity = 0.1, Route = `direct`, Skip Planner = `True`.
  - *Deep Reasoning*: Matches keywords (`architecture`, `debug`, `refactor`, `complex`). Base Complexity = 0.9, Route = `premium`, Skip Planner = `False`.
  - *Agentic*: Matches keywords (`create`, `write`, `plan`, `execute`) or exact match "do it"/"go". Base Complexity = 0.6, Route = `planner`, Skip Planner = `False`.
  - *Chat (Default)*: Base Complexity = 0.4, Route = `mid-tier`, Skip Planner = `True`.
- **Complexity Modifiers:** Additions made to base complexity score:
  - `word_count > 200`: +0.2
  - `multi_part` detected (lists, bullets, "and"): +0.15
  - `has_code` detected: +0.1
  - `tech_density > 0.15` (15% of words are technical terms): +0.05
  - `conditional_count >= 2` ("if", "when", "unless"): +0.05
- **Constraint**: Final complexity score is clamped between 0.0 and 1.0.
**Outputs**: Dictionary of routing flags: `class`, `complexity`, `route`, `skip_planner`, `estimated_tokens`, `needs_reasoning`, `needs_tools`, `context_weight`.
**Dependencies**: Heuristic keyword sets (`_REFLEX_KEYWORDS`, `_DEEP_KEYWORDS`, `_AGENTIC_KEYWORDS`, `_CONDITIONAL_WORDS`, `_TECHNICAL_TERMS`) (`core/controller/complexity_scorer.py`).

## 3. Global State Machine Transitions
**Description**: Enforces valid system states across all autonomous flows. Prevents illegal execution flows (e.g., executing without approval).
**Trigger**: Components requesting a state transition via `transition(new_state)`.
**Conditions**:
- Validates requested target state against the `_ALLOWED_TRANSITIONS` matrix.
- Key transitions:
  - `IDLE` → `THINKING`, `PLANNING`, `LISTENING`, `SHUTDOWN`
  - `PLANNING` → `RISK_EVALUATION`, `REVIEWING`, `IDLE`, `ERROR`, `SPEAKING`
  - `RISK_EVALUATION` → `AWAITING_CONFIRMATION`, `APPROVED`, `CANCELLED`, `ACTING`, `IDLE`, `ERROR`
  - `EXECUTING` → `REFLECTING`, `SPEAKING`, `COMPLETED`, `IDLE`, `ERROR`, `ABORTED`
**Outputs**: If valid, sets `_state = new_state`, logs the transition to the audit trail, and fires event bus listeners. If invalid, raises `IllegalTransitionError` and logs a failure audit entry.
**Dependencies**: `State` Enum, system event bus, concurrency locks (`core/state_machine.py`).

## 4. Autonomy Governance (Permissions)
**Description**: Dynamically enforces execution permissions without relying on hardcoded tool whitelists.
**Trigger**: Prior to any tool execution invocation by the agent.
**Conditions**: Evaluates the requested tool against the current `AutonomyLevel`:
- **Level 0 (CHAT_ONLY)**: Returns `False`. Tool execution is disabled.
- **Level 1 (SUGGEST_ONLY)**: Returns `False`. Tool execution is blocked (suggestions only).
- **Level 2 (READ_ONLY)**: Returns `True` only if `is_write_tool()` resolves to `False`. Blocked otherwise.
- **Level 3 (WRITE_WITH_CONFIRM)**: Returns `True` for all allowed tools, but flags write operations with `requires_confirmation = True`.
- **Level 4 (AUTONOMOUS)**: Returns `True` for all allowed tools. `requires_confirmation = False`.
- *Write Determination*: Checks dynamic memory (`write_tools`, `read_only_tools`), capability attributes, or falls back to risky string heuristics (`write`, `delete`, `launch`, `click`, etc.).
- *Unknown tools*: Blocked by default unless dynamically registered.
**Outputs**: Returns a tuple `(allowed: bool, reason: str)`.
**Dependencies**: Tool Registry, `AutonomyLevel` enum (0-4) (`core/autonomy/autonomy_governor.py`).

## 5. Risk Evaluation Matrix
**Description**: Deterministic risk classification for planned tool actions. Assesses danger level before reaching the governor.
**Trigger**: A list of planned actions submitted to `evaluate()` or `evaluate_plan()`.
**Conditions**: 
- Iterates through actions checking dynamic registries and config files.
- If not explicitly mapped, categorizes via safe keyword heuristics:
  - **CRITICAL**: Contains `shell`, `exec`, `subprocess`, `delete_file`, `format_disk`, `wipe_disk`, etc.
  - **HIGH**: Contains `spawn`, `pip_install`, `env_write`, `system_config`.
  - **CONFIRM**: Contains `write`, `launch`, `click`, `turn_on`, `drag`, `hotkey`.
  - **MEDIUM**: Contains `read`, `capture`, `sensor`, `search`.
  - **LOW**: Everything else.
**Outputs**: Calculates a `RiskResult` containing `max_level` (LOW, MEDIUM, CONFIRM, HIGH, CRITICAL) and lists of specific blocking, high risk, or confirming actions. If `CRITICAL`, execution is typically blocked entirely.
**Dependencies**: `[risk]` configurations, Capability Registry attributes (`core/autonomy/risk_evaluator.py`).

## 6. Goal Scheduling & Retry Backoff
**Description**: Handles delayed execution and exponential back-off rules for queued agent missions.
**Trigger**: Dispatched periodically (`scheduler.due()`), or triggered when a mission fails and invokes a retry.
**Conditions**:
- **Due calculation**: `utcnow() >= run_at` AND status is `WAITING`.
- **Retry Logic**: If `attempt_number < max_attempts`, reschedule. If exceeded, return False (cancel).
- **Delay formula**: `delay = base_delay_seconds * (backoff_factor ** (attempt_number - 1))`
**Outputs**: Returns a list of `ScheduledMission` entries. Pushes attempts counter up and sets future `run_at` stamps.
**Dependencies**: Task queue, `datetime.now(timezone.utc)` (`core/autonomy/scheduler.py`).

## 7. Explicit Web Search Intent Mapping
**Description**: Skips LLM planning steps when users issue explicit web lookup commands.
**Trigger**: Processing raw user queries via `IntentRouter`.
**Conditions**: 
- Validates if query string contains explicit markers (e.g., "search the web", "browse the internet", "google it").
- Validates active window requests (e.g., "what app is active").
**Outputs**: If explicit web is detected, intercepts the loop, transitions dashboard to `EXECUTE`, calls `handle_web_search()` immediately, stores the turn in `HybridMemory`, and returns.
**Dependencies**: `WEB_SEARCH_EXPLICIT_PHRASES` list, `is_explicit_web_search()` helper (`core/controller/request_rules.py`, `core/controller/intent_handlers.py`).

## 8. Runtime Dependency Crash Protection
**Description**: Business rule handling missing dependencies and circular imports gracefully rather than crashing.
**Trigger**: Start up routine `protect_runtime` and `safe_import()`.
**Conditions**:
- AST scans codebase for unresolved `import` and `from X import Y`.
- When an import fails dynamically (`ModuleNotFoundError`), the fallback handler triggers.
**Outputs**: Returns a mocked `FallbackMock` proxy object that logs errors instead of hard failing.
**Dependencies**: Python `ast`, `importlib.util` (`core/runtime/import_validator.py`).
