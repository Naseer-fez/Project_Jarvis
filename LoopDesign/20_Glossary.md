# 20. Glossary & System Ontology Engine

## 1. System Intent & Rationale (WHY it exists)
This document serves as the definitive Semantic Authority and Ontology Engine for the Jarvis AI Operating System Framework. In complex, multi-agent LLM systems, semantic drift—where developers, agents, and documentation use conflicting terms for the same memory structure or control flow—leads to catastrophic prompt misalignment. This glossary exists to enforce absolute conceptual rigidity, ensuring that every architectural abstraction, system prompt, and codebase reference relies on a unified, unambiguous vocabulary.

## 2. Core Responsibilities (WHAT it owns)
- **Conceptual Standardization:** It owns the strict definitions of all subsystems, state machines, memory structures, and runtime behaviors.
- **Prompt Anchoring:** It acts as the foundational dictionary injected or referenced during meta-prompt engineering, guaranteeing that LLMs do not hallucinate system capabilities.
- **Architectural Boundary Definition:** By strictly defining what a component *is*, it implicitly defines what it *is not*, preventing scope creep and unauthorized cross-component coupling.
- **Physical Module Mapping:** Binding abstract concepts directly to their physical module counterparts (e.g., `core/autonomy/autonomy_governor.py`) to eliminate ambiguity.
- **Programmatic Schemas:** Storing the exact datatypes (e.g., `EventRecord`) and valid transition matrices to ensure clean-room reconstruction works flawlessly.

## 3. System Interactions (HOW it interacts)
While a static document in human terms, the Glossary acts as the semantic schema for the entire framework. It interacts directly with the `LLM Orchestrator` and `Agentic Loop` by dictating the specific JSON schema keys, event types, and trace terminologies used in the codebase. Every system prompt, risk evaluation heuristic, and logging mechanism uses the exact nomenclature established here.

## 4. Cascading Effects (WHAT breaks if removed)
If the system ontology is removed or ignored:
- **Semantic Drift & Hallucination:** Agents will invent their own terminology for internal states, failing to trigger registered tools.
- **State Machine Collapse:** The disconnect between intended states (e.g., `AWAITING_CONFIRMATION`) and hallucinated states (e.g., `PENDING_USER`) will orphan execution threads.
- **Reconstruction Failure:** Without a definitive map of concepts, any clean-room reconstruction attempt will incorrectly recreate boundaries, leading to fatal race conditions and broken dependency injection graphs.

## 5. Clean-Room Reconstruction Directive (HOW to rebuild)
To rebuild the Ontology Engine from scratch without source code:
1. **Extract Subsystem Boundaries:** Analyze the raw capabilities (e.g., text generation, database writes, OS commands) and cluster them into logical nodes.
2. **Assign Rigid Nomenclature:** Define immutable terms for the control flow (`Agentic Loop`), data persistence (`Relational Memory`, `Semantic Memory`), and routing (`Event Bus`).
3. **Map State Transitions:** Explicitly define the vocabulary and exact matrix for state transitions (e.g., `_ALLOWED_TRANSITIONS`).
4. **Enforce Prompt Consistency:** Mandate that all system prompts (`REFLECT_SYSTEM_PROMPT`, etc.) are entirely rewritten to strictly adhere to this standardized vocabulary.
5. **Implement Exact Schemas:** Use the precise schemas defined below for all core payloads.

---

## System Ontology & Terminology

### A. Core Control Flow & Orchestration

**Agentic Loop (Loop Engine)**  
*Physical Domain:* `core/agent/agent_loop.py`  
The primary asynchronous control flow structure that transforms passive LLM text generation into proactive OS manipulation. It rigorously executes the `plan -> risk -> confirm -> execute -> reflect` lifecycle to resolve complex, multi-step tasks.

**Autonomy Governor (Risk Evaluator)**  
*Physical Domain:* `core/autonomy/autonomy_governor.py` and `core/autonomy/risk_evaluator.py`  
The definitive security sandbox. It intercepts every planned action from the DAG Executor, calculates an empirical risk score, and halts automated execution (entering the `AWAITING_CONFIRMATION` state) if the threshold exceeds the currently authorized autonomy level.

**DAG Executor**  
*Physical Domain:* `core/executor/engine.py`  
The underlying Directed Acyclic Graph planner. It translates high-level natural language goals into a structured matrix of non-dependent tool calls, allowing for concurrent `asyncio` execution of disjointed tasks.

**LLM Orchestrator (Dispatcher)**  
*Physical Domain:* `core/llm/model_router.py`  
The dynamic model-routing subsystem. It intercepts standardized user intents and routes them to tiered models (e.g., local 7B models for reflexive tasks, cloud 70B models for reasoning) to optimize latency, API cost, and token limits.

**Facade Controller (JarvisControllerV2)**  
*Physical Domain:* `core/controller_v2.py`  
The central traffic director. A strict implementation of the Facade design pattern that decouples front-end interfaces (Voice, CLI) from back-end logic, subscribing to intents via the Event Bus and dispatching them into the framework.

**State Machine**  
*Physical Domain:* `core/state_machine.py`  
The global lifecycle tracker governing agent operations. It enforces strict, atomic transitions between predefined phases: `IDLE`, `LISTENING`, `THINKING`, `EXECUTING`, and `AWAITING_CONFIRMATION`.  

*Exact Transition Schema:*
```python
_ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.THINKING, State.PLANNING, State.LISTENING, State.SHUTDOWN},
    State.THINKING: {State.IDLE, State.PLANNING, State.ERROR},
    State.PLANNING: {State.RISK_EVALUATION, State.REVIEWING, State.IDLE, State.ERROR, State.SPEAKING},
    State.RISK_EVALUATION: {State.AWAITING_CONFIRMATION, State.APPROVED, State.CANCELLED, State.ACTING, State.IDLE, State.ERROR},
    State.AWAITING_CONFIRMATION: {State.APPROVED, State.CANCELLED, State.ACTING, State.IDLE, State.ERROR},
    State.APPROVED: {State.EXECUTING, State.ACTING, State.IDLE, State.ERROR},
    State.ACTING: {State.OBSERVING, State.IDLE, State.ERROR},
    State.OBSERVING: {State.ACTING, State.REFLECTING, State.IDLE, State.ERROR},
    State.REFLECTING: {State.SPEAKING, State.IDLE, State.ERROR, State.COMPLETED},
    State.REVIEWING: {State.EXECUTING, State.ABORTED, State.IDLE, State.ERROR},
    State.EXECUTING: {State.REFLECTING, State.SPEAKING, State.COMPLETED, State.IDLE, State.ERROR, State.ABORTED},
    State.COMPLETED: {State.IDLE, State.SHUTDOWN},
    State.CANCELLED: {State.IDLE, State.SHUTDOWN},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.ERROR, State.COMPLETED},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.ERROR},
    State.TRANSCRIBING: {State.PLANNING, State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE, State.SHUTDOWN},
    State.ABORTED: {State.IDLE, State.SHUTDOWN},
    State.SHUTDOWN: set(),
}
```

**LIFO Topological Rollback**  
*Physical Domain:* Built into execution flows handling failure recovery.  
The fail-safe recovery protocol. Executing within a shielded `asyncio` context, it leverages Last-In-First-Out logic to cleanly revert partial state mutations if a multi-step DAG execution fails or times out (e.g., 300s limit).

### B. Memory, State & Persistence

**Semantic Memory**  
*Physical Domain:* `core/memory/semantic_memory.py`  
The fuzzy, associative datastore powered by a Vector Database (ChromaDB). It maps continuous strings into high-dimensional float arrays to retrieve past episodes, contextual knowledge, and code snippets based on cosine similarity.

**Relational Memory**  
*Physical Domain:* `core/memory/sqlite_storage.py`  
The rigid, deterministic SQLite datastore operating in WAL mode. It acts as the immutable ledger for explicit configuration overrides, session histories, and precise Boolean audits of tool executions.

**Persona Model (`user_profile.json`)**  
*Physical Domain:* Configuration files and `core/profile.py`  
The system’s internal psychological and demographic anchor. A mutable configuration file tracking the operator's timezone, technical competence, and interaction metrics to enforce behavioral alignment across sessions.

**Context Compressor**  
*Physical Domain:* `core/memory/context_compressor.py`  
The background token-management thread. It continuously monitors the sliding window of conversational history, triggering an LLM summarization call when limits are breached, flushing raw text, and committing the summary to long-term Semantic Memory.

### C. Execution & Interaction Primitives

**Dependency Injection (DI) Container (ServiceContainer)**  
*Physical Domain:* `core/runtime/container.py`  
The foundational instantiation registry. No component manually creates another; the DI container resolves all dependencies at runtime, ensuring complete modularity and testability.

**Event Bus**  
*Physical Domain:* `core/runtime/event_bus.py`  
The central asynchronous Pub/Sub messaging pipeline. Components communicate exclusively by emitting and subscribing to strictly typed `EventRecord` dataclasses, effectively decoupling producers from consumers.

*Exact EventRecord Schema:*
```python
@dataclass(frozen=True)
class EventRecord:
    """Replayable event envelope stored by the local event bus."""
    event_id: str
    event_type: str
    payload: Any
    source: str = "runtime"
    created_at: float = field(default_factory=time.time)
```

**ExecutionTrace**  
*Physical Domain:* `core/agent/agent_loop.py`  
The exhaustive telemetry payload. A strictly typed object capturing the full anatomical history of an agent loop's lifecycle, including goals, iterations, raw `<think>` blocks, risk scores, and final reflections.

*Exact ExecutionTrace Schema:*
```python
@dataclass
class ExecutionTrace:
    goal: str
    iterations: int = 0
    plan: Optional[dict[str, Any]] = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    risk_scores: list[dict[str, Any]] = field(default_factory=list)
    think_blocks: list[str] = field(default_factory=list)
    reflection: Optional[str] = None
    final_response: str = ""
    success: bool = False
    stop_reason: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
```

**Capability Registry (Tool Registry)**  
*Physical Domain:* `core/registry/registry.py`  
The declarative bridge between semantic requests and executable code. It dynamically maps LLM-requested function names and JSON schemas to validated, callable Python functions.

**ToolObservation**  
*Physical Domain:* `core/capability/base.py`  
The standardized output payload derived from executing a capability. It is rigorously truncated (e.g., capped at 4000 characters) before being fed back into the orchestrator, neutralizing the threat of context-window poisoning or ReDoS attacks.

*Exact ToolObservation Schema:*
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
