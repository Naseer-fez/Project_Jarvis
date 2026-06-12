# 02 Architecture

## Overview
Jarvis employs a modular, dependency-injected architecture designed for extensibility, loose coupling, and robust error recovery. The overarching pattern is a Facade-driven setup orchestrated by a central Dependency Injection (DI) container, complemented by an asynchronous Event Bus for cross-module communication.

This document performs deep semantic extraction of the core architectural subsystems, defining their intent, interactions, and reconstruction requirements.

---

## 1. Dependency Injection (DI) Container & Runtime Bootstrapper

### WHY does this subsystem exist?
A multi-modal AI system encompasses dozens of heavily interdependent components (e.g., memory needs the LLM to embed text; the LLM needs memory for context; tools need memory to read files). Hardcoding these instantiations leads to brittle, circular dependencies and prevents seamless swapping of implementations (e.g., swapping local LLMs for cloud endpoints or mocking tools during tests). The DI Container exists to centralize lifecycle management, strictly enforce interface contracts, and decouple domain logic from instantiation boilerplate.

### WHAT responsibility does it own?
- **Service Registration & Resolution:** Maintains a registry of singleton and transient services mapped by string identifiers or protocol types.
- **Lifecycle Management:** Controls the order of initialization during the boot sequence, ensuring base dependencies (like logging and Event Bus) are spun up before complex consumers (like the LLM Dispatcher).
- **Inversion of Control (IoC):** Provides a mechanism (`ServiceContainer.resolve()`) for components to request dependencies at runtime without knowing their concrete implementations.

### HOW does it interact with the rest of the system?
- The `Runtime Bootstrapper` (`main.py` -> `core.runtime.entrypoint`) creates the `ServiceContainer`.
- Subsystems like `JarvisControllerV2` query the container to retrieve `MemorySubsystem`, `LLMOrchestrator`, etc.
- Dependencies are frequently injected via `__init__` methods or dynamically resolved during tool execution to avoid circular imports.

### WHAT would break if removed?
- The entire application would fail to initialize due to recursive import loops and undefined initialization orders.
- Implementing alternate execution environments (e.g., headless mode vs. GUI dashboard mode) would require massive conditional logic duplicated across the codebase.
- Testing would become impossible without spinning up the entire actual infrastructure (SQLite, ChromaDB, active LLM APIs).

### HOW would it be rebuilt from scratch without source code?
1. **Define a Registry:** Create a thread-safe dictionary mapping service interfaces (or string names) to factory functions or instantiated objects.
2. **Topological Sort Initialization:** Implement a dependency graph solver that reads the required dependencies of each service and initializes them in a valid sequence.
3. **Locator Pattern:** Expose a global or context-bound `get_service(type)` method that dynamically retrieves instances, raising explicitly defined `DependencyResolutionError` exceptions if a required service is missing or uninitialized.

---

## 2. Event Bus (Pub/Sub)

### WHY does this subsystem exist?
An agentic loop is highly asynchronous. While the LLM is thinking, the UI needs to spin, background goals might trigger, and telemetry needs to be recorded. Direct method calls between these disparate domains create rigid coupling. The Event Bus exists to decouple the producers of state changes from the consumers of those changes, enabling a reactive architecture.

### WHAT responsibility does it own?
- **Asynchronous Message Passing:** Facilitates fire-and-forget message broadcasting.
- **Topic Subscription:** Allows components to subscribe to specific event types (e.g., `SYSTEM_ERROR`, `STATE_TRANSITION`, `LLM_STREAM_CHUNK`).
- **Telemetry & Audit Forwarding:** Acts as the primary pipeline funneling transient runtime data to permanent logs or the user dashboard.

### HOW does it interact with the rest of the system?
- The `StateMachine` emits transition events to the bus.
- The `LLMOrchestrator` emits token chunks.
- The `DashboardRuntime` (if enabled) subscribes to the bus to push WebSockets updates to the UI.
- The `AutonomyGovernor` listens for high-risk tool execution requests broadcasted over the bus.

### WHAT would break if removed?
- The system would lose all real-time UI/CLI feedback during long-running LLM operations.
- Background tasks (like the `GoalManager`) would have no safe way to notify the main controller to interrupt the user or speak via TTS.
- Telemetry and audit logging would require hardcoded hooks sprinkled throughout hundreds of functions.

### HOW would it be rebuilt from scratch without source code?
1. **Message Envelope:** Define an `EventRecord` dataclass containing `timestamp`, `event_type` (Enum), `payload` (Dict), and `source` (String).
    ```python
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class EventRecord:
        event_id: str
        event_type: str
        payload: Any
        source: str
        created_at: float

        def to_dict(self) -> dict[str, Any]: ...
    ```
2. **Broker:** Create a centralized `EventBus` class using `asyncio.Queue` or a list of callback functions.
3. **Routing:** Implement an `emit(event)` method that non-blockingly distributes the event to all registered `subscribe(event_type, callback)` handlers.
4. **Error Isolation:** Ensure that if a subscriber's callback throws an exception, it is caught and logged by the Event Bus, preventing a cascading failure that crashes the publisher.

---

## 3. LLM Orchestrator & Dispatcher

### WHY does this subsystem exist?
Not all queries require a massive, expensive, and slow 70B parameter model. Simple classification (e.g., "Is this a tool request or chat?") can be handled by a fast local 7B model, while complex architecture generation requires a high-tier reasoning model. The LLM Orchestrator exists to optimize token cost, latency, and capability matching by dynamically routing prompts to the most appropriate intelligence tier.

### WHAT responsibility does it own?
- **Model Abstraction:** Normalizes APIs across different providers (Ollama, Gemini, Groq, Anthropic) into a unified `generate_response()` and `stream_response()` interface.
- **Tiered Routing:** Classifies tasks into categories (Reflexive, Reasoning, Execution) and assigns them to the correct model backend.
- **Context Window Management:** Truncates, summarizes, or rejects prompts that exceed the token limits of the selected model.
- **Fallback Execution:** Automatically retries failing LLM calls or degrades to a secondary model if the primary API endpoint is unreachable.

### HOW does it interact with the rest of the system?
- Receives unified user intents from the `JarvisControllerV2`.
- Queries the `MemorySubsystem` to inject conversational history and semantic context into the system prompt.
- Returns structured JSON or raw text to the `AgentLoopEngine` or `TaskPlanner` for parsing.

### WHAT would break if removed?
- The system would lose all semantic reasoning capabilities, degrading into a standard hardcoded script runner.
- If reduced to a single hardcoded model without orchestration, the system would either be too slow/expensive for basic tasks or too incapable for complex tasks.

### HOW would it be rebuilt from scratch without source code?
1. **Provider Adapters:** Write standard interfaces (e.g., `BaseLLMProvider`) with implementations handling specific REST API protocols for different AI providers.
2. **Router Logic:** Implement a mapping configuration tying "Task Complexity Scores" to specific Model IDs.
3. **Prompt Pipeline:** Build a templating engine (like Jinja2) to assemble system instructions, tool schemas (JSON Schema), and user queries into a final string payload.
4. **Resilience Layer:** Wrap all outgoing HTTP calls in a retry circuit breaker using exponential backoff to handle rate limits and transient network failures.

---

## 4. Memory Subsystem

### WHY does this subsystem exist?
An LLM is inherently stateless. To maintain a coherent persona, remember past user preferences, track ongoing projects, and operate across system reboots, the agent requires an externalized memory structure. It needs to combine exact relational data (for configuration) with fuzzy semantic data (for concept retrieval).

### WHAT responsibility does it own?
- **Short-Term Context:** Maintains the sliding window of the current conversational transcript.
- **Episodic/Semantic Memory:** Interfaces with a Vector Database (ChromaDB) to embed and retrieve past knowledge, code snippets, or documentation based on cosine similarity to the current query.
- **Relational State:** Manages SQLite databases for rigid structured data like user profiles, system configurations, and past tool execution logs.
- **Memory Consolidation:** Periodically summarizes old short-term context and commits the summary to long-term vector storage to prevent infinite context growth.

### HOW does it interact with the rest of the system?
- Provides the `LLMOrchestrator` with pre-computed context blocks to inject into system prompts.
- Fed new data continuously by the `Controller` after every user interaction or tool execution.
- Queried explicitly by the `AgentLoopEngine` when executing "search memory" tools.

### WHAT would break if removed?
- Jarvis would suffer from complete "goldfish memory." Every interaction would start from a blank slate.
- Context windows would overflow rapidly during long tasks, causing LLM API rejections or hallucination spirals as early task constraints are forgotten.

### HOW would it be rebuilt from scratch without source code?
1. **Vector Store Integration:** Set up a local vector database (like Chroma or FAISS) and a local embedding model (like `all-MiniLM-L6-v2`) to turn strings into float arrays.
2. **Relational Schema (Unified):** Create a SQLite database with strict unified UTC `CURRENT_TIMESTAMP` formats (resolving the split-brain `memory.db` vs `jarvis_memory.db` drift).
    ```sql
    CREATE TABLE facts (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        source      TEXT NOT NULL DEFAULT 'user',
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        metadata    TEXT NOT NULL DEFAULT '{}'
    );
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
    CREATE TABLE actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        result TEXT,
        success INTEGER NOT NULL DEFAULT 1,
        metadata TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ```
3. **State Payload Injection Schema (`user_profile.json`):** Must enforce this strict implicit schema to prevent second-order prompt injections before injecting into prompt context.
    ```json
    {
      "name": "string",
      "communication_style": "string",
      "expertise_level": "string",
      "preferred_topics": ["string"],
      "timezone": "string",
      "language": "string",
      "interaction_count": "integer",
      "first_seen": "YYYY-MM-DDTHH:MM:SS.mmmmmm",
      "last_seen": "YYYY-MM-DDTHH:MM:SS.mmmmmm"
    }
    ```
4. **Retrieval Pipeline:** Implement a RAG (Retrieval-Augmented Generation) function that takes an incoming query, embeds it, queries the top-K closest vectors, and concatenates the text results into a "Context" string block.
5. **Pruning Mechanism:** Build a background thread that monitors token count in the active session and triggers an LLM summarization call when the threshold is breached, saving the summary to DB and flushing the raw messages.

---

## 5. Agentic Loop & Tool Execution Pipeline

### WHY does this subsystem exist?
An LLM generating text is fundamentally passive. To achieve "agency," the system must act upon its environment, evaluate the results of those actions, and correct course if it failed. The Agentic Loop provides the control flow structure to map LLM intentions to actual executable Python code (Tools) and feed the output back for further reasoning.

### WHAT responsibility does it own?
- **Task Planning:** Decomposing a high-level user request into a Directed Acyclic Graph (DAG) of discrete tool calls.
- **State Machine Orchestration:** Moving the system rigidly through `PLANNING` -> `RISK_EVALUATION` -> `EXECUTING` -> `REFLECTING` states.
- **Tool Routing & Execution:** Mapping an LLM-requested function name and JSON payload to actual registered Python functions, executing them safely, and capturing `stdout`/`stderr`.
- **Autonomy Governance:** Enforcing safety boundaries by halting execution in the `AWAITING_CONFIRMATION` state if a tool exceeds risk thresholds (e.g., `rm -rf`).

### HOW does it interact with the rest of the system?
- Invoked by the `JarvisControllerV2` once an intent is classified as requiring multi-step execution.
- Heavily relies on the `LLMDispatcher` to parse tool outputs and decide the next step.
- Triggers events on the `EventBus` for UI observation during execution.

### WHAT would break if removed?
- Jarvis could only converse; it could no longer manipulate the filesystem, run terminal commands, control the desktop, or perform API searches.
- Without the strict State Machine and Autonomy Governor, LLM hallucinations could instantly execute destructive OS commands without user intervention.

### HOW would it be rebuilt from scratch without source code?
1. **Tool Registry:** Create a system mapping tool names to callable Python functions, along with JSON Schemas defining their expected arguments.
2. **Explicit Failure Schema:** Tools and integrations must trap exceptions and normalize them into a rigid schema: `{"success": False, "data": None, "error": "Explicit error message"}`.
3. **Execution Trace Schemas:**
    ```python
    from dataclasses import dataclass
    from typing import Any, Optional

    @dataclass
    class ToolObservation:
        tool_name: str
        arguments: dict
        execution_status: str
        output_summary: str
        error_message: Optional[str]
        duration_seconds: float
        metadata: Optional[dict[str, Any]]

    @dataclass
    class ExecutionTrace:
        goal: str
        iterations: int
        plan: Optional[dict[str, Any]]
        observations: list[dict[str, Any]]
        risk_scores: list[dict[str, Any]]
        think_blocks: list[str]
        reflection: Optional[str]
        final_response: str
        success: bool
        stop_reason: str
        started_at: float
        ended_at: Optional[float]
    ```
4. **ReAct/Plan-and-Solve Loop:** Implement a `while not task_complete:` loop. Inside: 
   - Ask LLM for next action.
   - Parse action JSON.
   - Run requested Python function in a `try/except` block.
   - Append output (or traceback) to the context history.
5. **Safety Sandbox:** Implement an interceptor pattern before executing any tool. If the tool is flagged as `HIGH_RISK`, pause the loop, queue a prompt to the user, and wait for asynchronous approval before proceeding.
6. **DAG Executor:** (For advanced usage) Implement a topological sorter to execute non-dependent tools concurrently via `asyncio.gather`.
