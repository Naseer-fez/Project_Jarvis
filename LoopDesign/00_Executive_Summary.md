# 00 Executive Summary: Jarvis Autonomous OS Reconstruction Blueprint

## 1. WHY does this subsystem exist? (System Intent)
The Jarvis Autonomous OS exists to bridge the semantic reasoning of Large Language Models (LLMs) with high-privilege execution environments (local OS, desktop UI, background services, and external web properties). It exists not merely as a conversational chatbot, but as an **Agentic Operating System** capable of long-running, asynchronous, and proactive multi-step workflows. Its fundamental purpose is to abstract chaotic environments (like screen parsing, web scraping, and file system traversal) into deterministic, graph-based execution pipelines.

## 2. WHAT responsibility does it own? (Business Logic & Core Mandate)
The system owns the complete lifecycle of goal decomposition, execution, state retention, and failure recovery. Its responsibilities are explicitly divided across five core domains:
- **Orchestration & Routing (`JarvisControllerV2`, `IntentRouter`):** Triaging inputs and determining if an intent requires a deterministic fast-path or multi-step LLM reasoning.
- **Persistent State & Memory (`MemorySubsystem`):** Maintaining relational state (SQLite), semantic context (ChromaDB), and JSON-based execution graphs.
- **Proactive Execution (`AutomationManager`, `GoalRunner`):** Running headless task loops that monitor environments (watchdogs) and execute delayed goals without user prompting.
- **Safety Governance (`RiskEvaluator`):** Acting as the critical boundary between LLM reasoning and bare-metal execution, enforcing required human-in-the-loop (HITL) confirmations based on heuristic risk scoring.
- **Environment Manipulation (`Tools`, `Desktop`, `Voice`):** Mutating the host operating system, observing active applications, and handling synthetic I/O.

## 3. HOW does it interact with the rest of the system? (Workflows & Transitions)
Jarvis operates on a Facade-driven, Event-Sourced architecture. The core `ServiceContainer` injects dependencies into the `JarvisControllerV2`, which listens for inputs via CLI or the continuous `VoiceLayer`.
Upon receiving a stimulus:
1. **Introspection:** The system classifies the stimulus using an `llm_model_router` (e.g., routing complex reasoning to cloud LLMs, and reflexive UI queries to local Ollama models).
2. **State Transition:** State changes are captured via an asynchronous `EventBus` and persisted into the dual-layer storage systems (`jarvis_memory.db` and state machine JSON trackers like `automation_state.json` or `goals.json`).
3. **Execution Pipeline:** Multi-step workflows are delegated to the `DAGExecutor` (Directed Acyclic Graph), managing dependencies between sub-tasks and triggering tool bindings.

*(CRITICAL ADVERSARIAL DISCLOSURE: The current architecture suffers from severe state-transition conflicts. Async functions routinely collide with `threading.RLock` primitives, risking deadlock. Perfect reconstruction must implement unified async-safe locks and atomic `.tmp` swapping for all JSON state transitions).*

## 4. WHAT would break if removed? (Criticality)
If the primary Orchestration & Memory subsystems were removed, the entire autonomy architecture collapses:
- The **Agentic Loop** would lose long-term episodic memory, suffering from "split-brain" amnesia between sessions.
- Background tasks (like `GoalRunner` and `live_automation`) would instantly crash due to unbounded state loops losing their pointer references.
- The system would logically regress into a stateless, read-eval-print (REPL) CLI wrapper.
- Removal of the `RiskEvaluator` drops the system into unrestricted God-Mode, guaranteeing prompt injections and catastrophic OS mutations due to implicit trust boundaries.

## 5. HOW would it be rebuilt from scratch without source code? (Reconstruction Directive)
To reconstruct the Jarvis AI OS from pure semantics without source code, an engineering team must follow this exact reverse-engineering sequence:

1. **Establish the Interface Contracts (The Implicit Schemas):** Ignore the absence of explicit schema definitions found in static analysis. Reverse-engineer the `**kwargs` from the tool endpoints and construct strict Pydantic data models for every intent, action, and JSON state file.

```python
class AutomationState(BaseModel):
    saved_at: datetime
    seen_fingerprints: list[str]

class Goal(BaseModel):
    id: str
    description: str
    status: str

class ScheduleItem(BaseModel):
    trigger_time: datetime
    action: str

class GoalsState(BaseModel):
    saved_at: datetime
    goals: list[Goal]
    schedule: list[ScheduleItem]
```
2. **Unify Concurrency & Storage (Resolve Schema Chaos):** Build a single DI `ServiceContainer` dictating a unified `asyncio.Lock` concurrency model. Eliminate the fragmented dual SQLite database model (`memory.db` vs `jarvis_memory.db`) in favor of a single WAL-enabled database, enforcing strict UTC timestamps.

```sql
CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE episodic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, category TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE conversation_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_input TEXT NOT NULL, assistant_response TEXT NOT NULL, session_id TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, result TEXT, success INTEGER NOT NULL DEFAULT 1, metadata TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE facts (key TEXT PRIMARY KEY, value TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'user', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, metadata TEXT NOT NULL DEFAULT '{}');
```
3. **Rebuild the Routing & Governance Layer (Enforce Safety):** Implement the `IntentRouter` and `RiskEvaluator` BEFORE attaching LLMs. This ensures destructive actions are blocked at the infrastructure level, mitigating the God-Mode Default risk identified during Red Team audits.

```python
@dataclass
class IntentRoute:
    condition: Callable[[str, str, Any], bool]
    handler: Callable[[str, str, Any], Awaitable[str | None]]

class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    CONFIRM = 2
    HIGH = 3
    CRITICAL = 4
    FORBIDDEN = 4

@dataclass(frozen=True)
class RiskResult:
    level: RiskLevel
    blocking_actions: list[str] = field(default_factory=list)
    confirm_actions: list[str] = field(default_factory=list)
    high_risk_actions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
```
4. **Implement Bounded Memory Loops (Prevent OOM):** The `DAGExecutor` and `AutomationManager` must be rebuilt with strict array truncation limits (O(1) caps) and jittered exponential backoffs to prevent Thundering Herd scenarios and unbounded resource collapse during topological rollbacks.
5. **Attach Prompt Envelopes & LLM Adapters:** Finally, map the extracted prompt templates (from `LoopDesign/Prompts/`) into execution boundaries, safely injecting the reconstructed context and tools to breathe life into the core logic loop.
