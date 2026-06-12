# 06 Dependency Map: System Integration & Component Topologies

## 1. System Intent & Existence (WHY)
The Dependency Map exists to strictly formalize the topological relationships, execution links, and synchronization boundaries between all Jarvis subsystems. In a multi-modal, asynchronous AI architecture, naive direct imports create fatal circular dependencies, and unmanaged state sharing leads to catastrophic deadlocks. This map provides the authoritative source of truth for component initialization order, Inversion of Control (IoC) bindings, and event-driven data flow, ensuring that concurrent agents do not fracture the unified system state.

## 2. Core Responsibilities (WHAT)
- **Topological Initialization Mapping:** Defining the exact sequence in which subsystems must boot to satisfy downstream requirements (e.g., Config $\rightarrow$ EventBus $\rightarrow$ Storage $\rightarrow$ Cognitive Engines $\rightarrow$ Controller).
- **Concurrency & Synchronization Boundaries:** Explicitly outlining which components share thread-unsafe states (like `goals.json`, `user_profile.json`, and SQLite connections) and defining the required lock hierarchies to prevent data corruption.
- **Implicit Dependency Tracking:** Documenting non-code dependencies, such as environment assumptions, filesystem constraints, and dynamic input schemas (`**kwargs`) that static analysis often misses.
- **Interface Contract Enforcement:** Acting as the blueprint for the Dependency Injection (DI) container, dictating the required protocols and strict boundaries between architectural layers.

## 3. Interactions & The Dependency Topology (HOW)
The system operates on a strictly layered, inverted dependency model. Inner layers (Core Primitives) cannot depend on outer layers (Features). Inter-layer communication must occur via the Event Bus or explicit dependency injection.

### Layer 0: Core Primitives (Zero External Dependencies)
- **Config & Environment (`ops.production`, `registry.registry`)**: Singletons containing the absolute truth of system configuration and capabilities.
- **Event Bus (`runtime.event_bus`)**: The asynchronous nervous system facilitating Pub/Sub communication.
- **Logger (`logging.logger`)**: Global sink for all telemetric, diagnostic, and audit output.
*Interaction:* Bootstrapped first. All higher-order components receive these via constructor injection.

### Layer 1: Persistence & State Management
- **Memory Storage (`memory.sqlite_storage`, `memory.hybrid_memory`, `memory.embeddings`)**:
  - *Depends on:* Layer 0 + strict OS filesystem boundaries (`jarvis.ini`).
  - *Dependency Caveat (Split-Brain Prevention):* Must strictly resolve as an application-wide Singleton. Dual instantiation will corrupt SQLite locks and fragment context. Must enforce strict UTC timestamp constraints across all reads/writes to prevent temporal drift.

### Layer 2: Cognitive & Control Engines
- **LLM Orchestrator (`controller.llm_orchestrator`, `llm.client`, `llm.model_router`)**:
  - *Depends on:* Layer 0 + Layer 1 (`MemoryStorage` for context retrieval and RAG).
  - *Interaction:* Ingests contextual memory, dynamically routes requests to local/cloud models, and yields text/JSON tokens to the Event Bus.
- **Execution DAG & State Machine (`executor.engine`, `executor.state_machine`)**:
  - *Depends on:* Layer 0 + Layer 1.
  - *Dependency Caveat (The Synchronization Paradox):* Requires unified concurrency locks. The State Machine must utilize `asyncio.Lock` across all transitions to prevent race conditions during concurrent multi-step rollbacks, explicitly banning the use of OS-bound thread locks (`threading.RLock`).

### Layer 3: Actuators & Interface Surfaces
- **Tool Registry & Execution (`tools.system_automation`, `tools.web_search`, `tools.hardware_tools`)**:
  - *Depends on:* Layer 0, Layer 1, OS APIs, and external Network Interfaces.
  - *Interaction:* Operates under strict Risk Governance assumptions. Actuators must never be directly imported by cognitive engines; they rely entirely on dynamic dispatch via the `Registry`.
- **Sensory Loops (`voice.voice_loop`, `runtime.entrypoint`)**:
  - *Depends on:* All lower layers.
  - *Interaction:* The outermost interaction shell. Constantly polls hardware (mic, keyboard, file drops), triggers intent generation, and blocks asynchronously until the Event Bus returns a synthesized operational response.

## 4. Failure Modes & Cascading Effects (WHAT BREAKS)
- **Circular Initialization Deadlocks:** If an actuator (e.g., `web_tools`) directly imports a cognitive engine (e.g., `llm_orchestrator`) instead of relying on the Event Bus, the Python import resolver will crash the runtime immediately upon startup, halting the entire operating system.
- **Concurrency Collapse:** If the shared dependencies on JSON state files (`automation_state.json`) or database cursors are not protected by unified `asyncio` locks, concurrent agent loops (e.g., proactive monitors vs. direct user commands) will overwrite each other. This results in split-brain memory, persistent state corruption, and immediate context fragmentation.
- **Unbounded Resource Exhaustion (The Happy-Path Fallacy):** If components do not respect the strict timeout and memory boundary architectural constraints of their dependencies (e.g., an unbounded array `seen_fingerprints` passed between tools and memory), the central rollback mechanism will fracture. This will spawn orphaned zombie tasks that continuously consume RAM until an Out-of-Memory (OOM) crash occurs.
- **God-Mode Prompts (The Safety Illusion):** Without strict schema enforcement between the User Profile memory state and the LLM Orchestrator, malicious payloads injected into `user_profile.json` act as second-order prompt injections, overriding systemic guardrails and granting arbitrary execution capabilities to the model.

## 5. Reconstruction Strategy (HOW TO REBUILD)
To perfectly reconstruct the dependency topology from scratch without access to the original source code, the following rigorous sequence must be employed:
1. **Initialize a Strict IoC (Inversion of Control) Container:** Build a central runtime bootstrapper (like `container.py`) that acts as the sole authorized instantiator of classes. Hardcoded instantiation of complex subsystems must be strictly prohibited.
2. **Implement Topological Sorting:** Programmatically define the boot sequence (`Config $\rightarrow$ EventBus $\rightarrow$ Storage $\rightarrow$ LLM $\rightarrow$ Tools $\rightarrow$ Controller`). The DI container must resolve dependencies from bottom-to-top, instantly panicking and halting on any cycle detection.
3. **Establish Cross-Boundary Contracts (Implicit Schemas):** Define the explicit shapes of all payloads passed between components. Assume nothing about `**kwargs`; if an LLM is expected to call a tool, the exact JSON schema it must adhere to must be defined and validated as a strict dependency contract.
4. **Enforce Atomic State-Locking Primitives:** Map out all filesystem states (JSON files and SQLite databases). Wrap every single read/write operation to these files in an atomic, unified async lock bound to the dependency container, guaranteeing thread-safe, ACID-compliant mutations across all asynchronous tasks.
5. **Implement Bounded Interfaces:** Guarantee that all loops, timeouts, and array structures passed between dependencies enforce explicit upper bounds, preventing cascading resource exhaustion.

## 6. Exact Programmatic Schemas (Dependency Contracts)

To resolve the "Synchronization Paradox" and enforce strict dependency inversion, the following exact programmatic schemas must be implemented and validated as strict dependency contracts across all inter-process communications and state-file mutations.

### User Profile State (`user_profile.json`)
```python
class UserProfile:
    name: str
    communication_style: str
    expertise_level: str
    preferred_topics: list[str]
    timezone: str
    language: str
    interaction_count: int
    first_seen: datetime
    last_seen: datetime
```

### Automation State (`automation_state.json`)
```json
{
  "saved_at": "string (ISO-8601)",
  "seen_fingerprints": ["string (hash)"]
}
```

### Capability Contract (`ToolObservation`)
```python
@dataclass
class ToolObservation:
    tool_name: str
    arguments: dict
    execution_status: str       # "success" | "failure"
    output_summary: str
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] | None = None
```

### Desktop Execution Contracts
```python
@dataclass(frozen=True)
class DesktopAction:
    action_type: DesktopActionType | str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expected_change: str = ""
    risk_tier: DesktopRiskTier | str | None = None
    requires_approval: bool | None = None
    action_id: str = field(default_factory=lambda: _new_id("act"))
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class DesktopActionResult:
    action_id: str
    action_type: str
    success: bool
    status: DesktopActionStatus
    output: str = ""
    error: str = ""
    risk_tier: DesktopRiskTier = DesktopRiskTier.MEDIUM
    audit_hash: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Memory Storage Schemas (`memory.db`)
```sql
CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    event TEXT,
    category TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    user_input TEXT,
    assistant_response TEXT,
    session_id TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    result TEXT,
    success INTEGER NOT NULL DEFAULT 1,
    metadata TEXT,
    timestamp TEXT NOT NULL
);
```

### Authentication Schemas (`auth.db`)
```sql
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_tokens (
    token_hash TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_used_at REAL
);
```
