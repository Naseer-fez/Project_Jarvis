# Error Handling & Resiliency Subsystem

## 1. System Intent: WHY does this subsystem exist?
The Error Handling subsystem exists to act as the final defensive boundary against the fundamental unreliability of autonomous agents. Given that Jarvis interacts with non-deterministic LLMs, volatile network boundaries, and asynchronous execution contexts, failure is an expected baseline, not an anomaly. This subsystem isolates faults, prevents partial execution from corrupting global state, and ensures that a single hallucination, timeout, or API rate limit does not permanently paralyze the background asyncio event loop or corrupt the user's desktop environment.

## 2. Core Responsibilities: WHAT responsibility does it own?
1. **Component-Level Containment:** Normalizes all integration and tool execution failures into a unified, safe schema (`IntegrationResult`).
2. **Sub-graph Transaction Management:** Coordinates LIFO (Last-In-First-Out) reverse-topological rollbacks within the DAG engine when asynchronous plan steps fail.
3. **Transient Fault Recovery:** Implements retry mechanics with exponential backoff for network-bound operations.
4. **State Machine Recovery:** Automatically maps native Python exceptions to explicit `ERROR` or `ABORTED` transitions within the rolling `StateMachine`.
5. **Circuit Breaking & Failsafes:** Enforces autonomy termination through `jarvis.ini` thresholds (`failsafe_auto_disable_on_error`, `failsafe_error_threshold`) to prevent infinite failure loops in headless modes.
6. **Process-Level Safety:** Intercepts fatal boot errors via `ImportValidator` and hooks uncaught background exceptions globally.

## 3. Workflow & Architecture: HOW does it interact with the rest of the system?

### 3.1 Tool & Capability Boundary (`IntegrationResult`)
Tools and integrations never return bare exceptions to the orchestrator. Every action executed via the `CapabilityRegistry` or `ToolRouter` is wrapped in an `execute()` method that catches internal exceptions (e.g., `subprocess.CalledProcessError`, IMAP injection attempts, geolocator failures) and normalizes them into a strict payload:

```python
IntegrationResult = dict[str, Any]

@dataclass
class ToolResult:
    """Standardised return type for all Jarvis tool functions."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""
```

This prevents cascading exceptions from blowing up the parent LLM dispatcher.

### 3.2 Execution Engine & DAG Rollbacks (`engine.py`)
The `PlanDAG` executor manages complex, multi-step actions. If a node fails:
1. **Retries:** The engine attempts to retry the step using a naive exponential backoff (`backoff *= 2.0`) up to a predefined `retry_count`.
2. **LIFO Rollback:** If the step permanently fails after retries, the engine walks backwards down the executed dependency graph. It invokes the explicit `rollback` function mapped to each successful prior step to undo partial side-effects (e.g., deleting a drafted email if the sending step failed).
3. **Task Abort:** A permanent failure of a step (or a rollback failure) fractures the graph, halting execution and returning `{"status": "failure", "error": "..."}`.

### 3.3 State Guard Context Managers (`state_machine.py` & `agent_loop.py`)
Transitions between agent states (e.g., `PLANNING` -> `EXECUTING`) are protected by an asynchronous `StateGuard` context manager (`__aenter__`/`__aexit__`). 
- **Unhandled Exceptions:** If an unhandled exception occurs inside the context, the `__aexit__` block automatically translates this into an `ERROR` transition.
- **Timeouts:** The `agent_loop.py` enforces a hard 5-minute execution bound (`asyncio.timeout(300)`). When this fires, the resulting `asyncio.CancelledError` is explicitly mapped to the `ABORTED` state, and the graph is immediately stopped.

### 3.4 Process & Initialization Failsafes (`bootstrap.py`)
At boot, Jarvis sets global exception handlers on both OS threads (`threading.excepthook`) and the async event loop (`loop.set_exception_handler`). This ensures unhandled async exceptions in background daemon tasks (like proactive monitoring) are logged to `jarvis.log` instead of silently terminating the task ("zombieing"). Furthermore, an `ImportValidator` catches `ModuleNotFoundError` and circular dependencies, disabling broken integrations without crashing the entire agent runtime.

#### Error Logging Structures

**JSONFormatter Envelope Schema (`core/logging/logger.py`):**
```python
envelope = {
    "timestamp": timestamp,
    "level": record.levelname,
    "trace_id": trace_id,
    "task_id": task_id,
    "component": record.name,
    "event": getattr(record, "event", "log_message"),
    "metadata": getattr(record, "metadata", {}) or {},
}
# if record.exc_text:
#     envelope["stack_trace"] = record.exc_text
```

**AuditLog Structure (`core/logging/logger.py`):**
```python
body = {
    "event_type": event_type,
    "payload": redacted_payload,
    "prev_hash": self._last_hash,
}
# Final record stored as body | {"hash": digest}
```

**Circuit Breaker Configuration (`jarvis.ini`):**
```ini
[risk]
failsafe_auto_disable_on_error = true
failsafe_error_threshold = 3
```

## 4. Dependencies & Weaknesses: WHAT would break if removed?
- **Immediate Crash on Latency:** Without the `IntegrationResult` containment, the first network timeout from Home Assistant or GitHub would throw an exception that bypasses the controller, crashing the master agent thread.
- **Orphaned System States:** Without LIFO DAG rollbacks, partial executions would corrupt real-world resources (e.g., charging a credit card without committing the DB record, rotating an API key but crashing before saving it).
- **Zombie Event Loops:** Without global exception hooks, an unhandled exception in a background thread would kill the thread silently, leaving Jarvis deaf to wake words or blind to screen updates while appearing "online."
- **Infinite Financial / API Drain:** Without the `failsafe_error_threshold` circuit breaker, a hallucinating agent in `LEVEL_4` autonomy could enter a tight retry loop, exhausting API credits or burning CPU indefinitely.

## 5. Clean-Room Implementation: HOW would it be rebuilt from scratch?
To reconstruct this subsystem without source code, an architect must follow these steps:
1. **Define the Containment Schema:** Implement an interface matching `IntegrationResult` (Success boolean, Data payload, Error string). Mandate that ALL tool integrations return this exact dictionary and never leak exceptions.
2. **Implement LIFO Graph Execution:** Build the executor so it stores an execution trace. If Node C fails, it must automatically call Node B's `.rollback()` and Node A's `.rollback()` sequentially. 
3. **Contextual State Boundaries:** Use `__aenter__` and `__aexit__` context managers around all agent iterations. If `exc_type is not None`, intercept it and transition the agent's database/state machine to a locked `ERROR` mode.
4. **Hard Process Limits:** Implement an overarching `asyncio.timeout(300)` wrapper around the agent execution loop to kill stuck LLM generation. Hook into the Python runtime's native `sys.excepthook` to prevent silent asyncio death.
5. **Mitigate Known Vulnerabilities (Adversarial Fixes):**
   - *Fix the Thundering Herd:* Ensure the exponential backoff implementation includes random jitter, otherwise retries will continuously trigger API rate-limit bans.
   - *Fix the Lock Mismatch:* Ensure the State Machine utilizes `asyncio.Lock` exclusively; utilizing `threading.RLock` alongside async contexts fundamentally breaks re-entrancy guarantees and causes deadlocks.
   - *Graceful Truncation:* Do not blindly truncate inputs at 4000 chars, as this silently destroys JSON syntax, causing subsequent unrecoverable parse failures.
   - *Timeout Propagation:* Ensure that the 5-minute task timeout is decoupled from the LIFO rollback execution. If the timeout cancels the rollback routine itself, the graph remains permanently fractured.
