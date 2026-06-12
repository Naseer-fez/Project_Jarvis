# 03 Runtime Behavior

## 1. WHY does this subsystem exist?
The Runtime Behavior subsystem acts as the central orchestration nervous system of the Jarvis agent. Because the system operates across highly non-deterministic LLM pipelines, real-world tool executions, and asynchronous user I/O (CLI, Voice, Dashboard), a rigorous runtime orchestration layer is required. It guarantees execution boundaries are enforced, contexts are strictly isolated, and catastrophic failures are systematically contained. It prevents the system from cascading into infinite cognitive loops and guarantees that state transitions occur precisely according to topologically defined allowed paths. 

## 2. WHAT responsibility does it own?
This subsystem owns the end-to-end application lifecycle and execution isolation:
- **Lifecycle Orchestration**: Handles bootstrapping (`entrypoint.py`), preflight health checks, daemon background server setups (Dashboard, `GoalRunner`), and graceful tear-downs via `SIGINT`/`SIGTERM` hooks.
- **Execution Confinement & Recursion Limits**: The `controller_v2.py` routes interactions, while the `dispatcher.py` strictly bounds execution loops, throwing fatal `RecursionError` if max depth (5) is breached, mitigating unbounded LLM runaway.
- **Topological State Management**: Manages strict state transitions (`IDLE` -> `THINKING` -> `PLANNING` -> `RISK_EVALUATION` -> `EXECUTING`) via a centralized `StateMachine`. It maintains a rolling audit trail of up to 100 transitions for traceback.
- **Context Isolation & Tracing**: Wraps every execution block within a `TaskExecutionContext` utilizing `contextvars` to inject `trace_id` for isolated logging and automatic pre/post-crash state snapshotting.
- **DAG Compilation & Engine Rollback**: Translates generated plans into Directed Acyclic Graphs (`dag.py`) via Kahn’s topological sort. Executes steps sequentially (`engine.py`) with support for replays (bypassing successful steps) and exponential backoff retries.
- **Asynchronous Task Automation**: A persistent `GoalRunner` daemon evaluates scheduler timeouts against saved persistent states (JSON dumps) to trigger background autonomous actions or Voice TTS alerts.

## 3. HOW does it interact with the rest of the system?
- The main `entrypoint.py` resolves user CLI flags to determine execution modes (Voice, Headless, CLI) and launches the `JarvisControllerV2`.
- When an event is triggered, the controller invokes the `AgentLoopEngine`, initiating the core execution cycle wrapped in an `asyncio.timeout(300)` bound.
- The `AgentLoopEngine` delegates plan generation to `TaskPlanner`, safety evaluation to `RiskEvaluator`, and user consent to the `AutonomyGovernor` (which pauses in `AWAITING_CONFIRMATION`).
- The `DAGExecutor` reads the sorted plan, interacts directly with the `ToolRouter` to execute external integration calls, and writes outcomes into the `TaskExecutionContext`'s memory map.
- The `StateMachine` acts as the shared backbone. When exceptions propagate, the context manager `StateGuard` intercepts `asyncio.CancelledError` (or similar faults) and forces the machine into `ABORTED` or `ERROR` states, emitting event bus signals globally.

## 4. WHAT would break if removed?
Removing this subsystem would result in total architectural collapse:
- The system would lack any unified entry point, completely preventing parallel dashboard servers or background scheduling from functioning.
- There would be no upper bound on execution paths; deep cyclic LLM generations would exhaust system memory and API limits immediately.
- A failure inside a complex multi-step action sequence would permanently corrupt external states (e.g., leaving dangling cloud resources) due to the absence of the DAG execution LIFO rollback capability.
- Context data would bleed across parallel executions because the isolation provided by `contextvars.Token` would be missing. Traceability and forensic crash reconstruction (`logs/traces/{trace_id}.json`) would disappear.
- Crucially, the capability to pause cognitive loops for user confirmation (`AWAITING_CONFIRMATION`) would not exist, allowing autonomous agents to execute devastating actions unilaterally.

## 5. HOW would it be rebuilt from scratch without source code?
To recreate the Runtime Behavior layer to identical functional parity:

### 5.1 Bootstrapping & Event Loops
Implement `entrypoint.py` executing `asyncio.wait(return_when=asyncio.FIRST_COMPLETED)` handling main user tasks parallel to graceful signal cancellation tasks. Configure early initialization hooks to scrub sensitive `.env` keys from crash logs to prevent secret leakage.

### 5.2 Strict State Machine
Build a thread-safe `StateMachine`. *Crucial constraint*: Do NOT mix OS-level `threading.RLock` with async routines (`asyncio.Lock`), as multiple async tasks share the same thread and bypass the RLock, leading to data races. Use `asyncio.Lock` universally. Enforce strict graph paths. Cap the audit trail explicitly and strip large data objects from memory dumps to prevent snapshot memory exhaustion.

### 5.3 Context & Trace Isolation
Build `TaskExecutionContext` utilizing `contextvars`. Ensure automatic serialization on scope exit. Mitigate unbounded file growth (such as in `seen_fingerprints`) by implementing pruning, and enforce atomic writes (`.tmp` swaps) for JSON persist files (`automation_state.json`, `goals.json`) to prevent data corruption during mid-write power failures.

### 5.4 Bounded Agent Execution Pipeline
Construct the execution `while` loop wrapped in an `asyncio.timeout(300)`. Beware of brittle regex matching for tags like `<think>`; use bounded parsers to avoid catastrophic ReDoS backtracking. Handle long inputs explicitly to avoid silent truncation (e.g., truncating JSON strings at 4000 characters leading to malformed payload crashes). Ensure that headless modes (`LEVEL_4` autonomy) are strictly bounded by configured financial or action-based budget constraints.

### 5.5 Resilient DAG Engine
Implement Kahn's topological sort for the execution graph. Execute steps using exponential backoff *with injected random jitter* to prevent thundering herd API retry storms. Implement strict LIFO reverse-topological rollbacks upon failure. Explicitly engineer the engine to await LIFO rollbacks even if the overarching 300-second `asyncio.timeout` triggers, and define fallback mechanisms if the rollback function itself throws an exception.

## 6. Exact Runtime Memory & Threading Schemas

### SQLite Memory Models (`memory.db` WAL mode)
```sql
CREATE TABLE preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE episodic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category TEXT
);

CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);
```

### State Persistence Schemas (JSON)
`automation_state.json`:
```json
{
  "saved_at": "ISO-8601 timestamp string",
  "seen_fingerprints": ["Array of SHA-256 strings"]
}
```

`goals.json`:
```json
{
  "saved_at": "ISO-8601 timestamp string",
  "goals": [],
  "schedule": []
}
```

### Threading & Execution Parameters
- **State Machine Synchronization**: Re-entrant thread-safe using `threading.RLock`. Do not use `asyncio.Lock`.
- **Audit Trail Limit**: Rolling audit trail capped at `100` elements.
- **Recursion Threshold**: `max_recursion_depth = 5` in the DispatchPipeline.
- **Max Loop Iterations**: `_DEFAULT_MAX_ITERATIONS = 10` in AgentLoopEngine.
- **Execution Confinement Bound**: Hardcoded task timeout via `asyncio.timeout(300)`.
- **Exponential Backoff Engine**: `backoff *= 2.0` up to `retry_count`.

### Execution Trace & DAG Schemas
**ExecutionTrace Object**:
```python
class ExecutionTrace:
    goal: str
    iterations: int
    plan: dict
    observations: dict
    risk_scores: dict
    think_blocks: list
    reflection: str
    final_response: str
    success: bool
    stop_reason: str
    timestamps: dict
```

**DAG Step Dict Schema**:
```python
{
    "id": str,
    "action": str,  # or "tool"
    "description": str,
    "params": dict,
    "retry_count": int,
    "rollback": dict,
    "depends_on": list
}
```
