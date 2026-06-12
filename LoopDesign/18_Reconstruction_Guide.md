# 18 System Reconstruction Guide

## 1. System Intent & Existence (WHY)
The Reconstruction Guide exists as the ultimate architectural fail-safe and master blueprint for the Jarvis Autonomous OS. It is designed to bridge the fatal gap between surface-level topological maps (AST parsing, raw strings) and the chaotic realities of runtime execution. It exists because "happy-path" documentation is insufficient for rebuilding an asynchronous, multi-agent AI framework. This guide captures the implicit contracts, hidden failure states, concurrency boundaries, and data constraints required to resurrect the system with 100% semantic fidelity from zero source code.

## 2. Core Responsibilities (WHAT)
This subsystem of documentation owns the responsibility of defining the **Rules of Engagement** for a clean-room rebuild. Its primary responsibilities include:
*   **Enforcing the Single Source of Truth:** Resolving schema contradictions (e.g., split-brain memory between `memory.db` and `jarvis_memory.db`) and dictating a unified standard for data persistence.
*   **Defining Implicit Execution Boundaries:** Mandating strict schemas for previously untyped `**kwargs` payloads, specifying exact JSON boundaries, and enforcing negative constraints on LLM prompts.
*   **Governing Concurrency Models:** Resolving the synchronization paradox by mandating strict asynchronous locks and atomic file operations, entirely replacing OS-thread naive implementations (`threading.RLock`).
*   11. **Managing the State-Collapse Prevention Protocol:** Defining how the system handles unbounded array growth, out-of-memory (OOM) scenarios, and LIFO topological rollback survivability when outer-loop timeouts trigger.
12. **Defeating the Error Schema Straitjacket:** Ensuring semantic error categorizations (e.g., `TransientNetworkError` vs `AuthFailure`) are preserved, rather than being flattened into generic `{"success": False}` JSON strings.
13. **Defining the Integration Result & Intent Mappings:** Explicitly binding inputs and outputs to strict dataclass definitions rather than loose strings, preventing context window bloat and parser hallucinations.

## 3. Interactions & Workflows (HOW)
The Reconstruction Guide acts as the meta-layer above the entire architectural documentation stack (Modules 01-17).
*   **Ingestion:** It consumes the raw system overviews, dependency graphs, and specifically the Red Team Adversarial Audits, synthesizing their uncovered vulnerabilities into actionable engineering directives.
*   **Orchestration of Rebuild:** It dictates the sequential workflow for the reconstruction engineering team, preventing premature component integration. Developers must consult this guide to understand how the `LLMOrchestrator` integrates with the `State Machine` without causing deadlocks.
*   **Validation:** During the rebuild, this document serves as the ultimate test matrix. A reconstructed module is only validated if it respects the failure modes, rollback conditions, and concurrency protections outlined herein.

## 4. Failure Modes & Cascading Effects (WHAT BREAKS)
If this guide is removed, ignored, or if reconstruction is attempted using only surface-level code analysis, the following catastrophic failures are guaranteed:
*   **Systemic Deadlocks:** Rebuilders will natively mix `asyncio` and thread-bound `RLock` primitives, causing the asynchronous event bus to permanently freeze during high-latency state transitions.
*   **State Corruption & Split-Brain Fragmentation:** The agent will concurrently mutate JSON configurations without atomic swaps, irrecoverably corrupting `user_profile.json` and `goals.json`. It will write to duplicate, conflicting SQLite schemas (`episodes` vs. `episodic_memory`), shattering its own continuous memory.
*   **Second-Order Prompt Injections:** Developers will fail to sanitize implicit inputs, allowing malicious payloads stored in the user profile to act as root-level prompt overrides, bypassing the Risk Evaluator entirely.
*   **Resource Collapse (Thundering Herds):** Exponential backoffs will be implemented without jitter, and automation arrays will scale unboundedly, resulting in memory exhaustion and self-inflicted DDoS attacks against external APIs.
*   **The LIFO Rollback Delusion:** A permanent failure on a non-compensatable step (e.g., sending an email) will fracture the graph without a two-phase commit schema or Dead Letter Queue, causing silent, unrecoverable state corruption.
*   **The Circuit Breaker Mirage:** Implementing error thresholds without a temporal sliding window will result in permanent service lockouts due to transient spikes, permanently bricking the agent without human intervention.

## 5. Reconstruction Strategy (HOW TO REBUILD FROM SCRATCH)
To achieve a 100% reconstruction without access to the original source code, the engineering team must follow this strict, chronological, clean-room directive:

### Phase 1: Environment, Primitives & Unified Concurrency
1.  **Define UTC-Enforced Time Standards:** Eliminate all timestamp drift. All SQLite databases and JSON schemas MUST use standardized ISO-8601 UTC formats. Lexicographical sorting is mandatory for memory coherence.
2.  **Implement Unified Concurrency Locks:** Ban `threading.RLock` in asynchronous contexts. Establish a unified `asyncio.Lock` model for all in-memory state transitions to resolve the explicit sync/async deadlock contradiction.
3.  **Atomic Persistence:** Mandate atomic `.tmp` swapping for all JSON file writes. No component may perform direct Read-Modify-Write on critical files (e.g., `automation_state.json`) without strict lock acquisition and rollback guarantees.

### Phase 2: Single-Source-of-Truth Memory Engine
1.  **Consolidate Relational Memory:** Discard the split-brain SQLite implementation. Build a singular `memory.db` in WAL-mode. The exact schema MUST be injected as:
    ```sql
    CREATE TABLE facts (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'user',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE episodic_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        category TEXT
    );
    CREATE TABLE conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT NOT NULL,
        assistant_response TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        session_id TEXT
    );
    CREATE TABLE actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        result TEXT,
        success INTEGER NOT NULL DEFAULT 1,
        metadata TEXT,
        timestamp TEXT NOT NULL
    );
    ```
2.  **Implement Bounded Context Windows:** Ensure all memory structures (both semantic ChromaDB and relational SQLite) enforce hard token truncation limits. Ensure `seen_fingerprints` and other arrays implement strict LIFO truncation to prevent O(N) memory scaling.
3.  **Define Implicit `**kwargs` Schemas:** Explicitly map the expected shape of dynamic payloads entering the `EventBus`. The exact schema for `EventRecord` MUST be:
    ```python
    @dataclass
    class EventRecord:
        event_id: str
        event_type: str
        payload: Any # Must be typed internally as a strict JSON dictionary
        source: str
        created_at: float
        def to_dict(self) -> dict[str, Any]: ...
    ```
4.  **Define Core State JSON Schemas:** The primary runtime state schemas must map perfectly to these Data Transfer Objects:
    ```python
    class UserProfile:
        name: str
        communication_style: str
        expertise_level: str
        preferred_topics: list[str]
        timezone: str
        language: str
        interaction_count: int
        first_seen: datetime # ISO-8601
        last_seen: datetime  # ISO-8601

    class GoalsState:
        saved_at: datetime   # ISO-8601
        goals: list[dict]
        schedule: list[dict]
    ```

### Phase 3: The State Machine & Robust Control Flow
1.  **Implement Timeout-Safe Rollbacks:** Build the LIFO reverse-topological rollback engine. If an action times out, the rollback must execute in an isolated, protected context to prevent state fracturing. Crucially, explicitly prevent asynchronous deadlocks by defining how a rollback operates if a 300s outer `asyncio.timeout` hits (e.g. by using secondary isolated timeouts, avoiding blind hangs on non-compensatable nodes).
2.  **Jittered Exponential Backoff & Semantic Errors:** All API clients and event-polling loops must implement randomized jitter within their backoff algorithms. Do not stringify errors. The Integration schema MUST NOT be a flat `{"success": False, "data": None, "error": "string"}` dictionary. It must retain semantic exception types (`TransientNetworkError` vs `AuthFailure`) to dictate transient retries vs terminal aborts.
3.  **Circuit Breaker Protocol:** Implement error limits utilizing a temporal sliding window (X errors per Y minutes). Include half-open probing to gracefully recover headless autonomy instead of permanently locking the system after an arbitrary raw count.

### Phase 4: Prompt Architecture & Binding Contracts
1.  **Strict Interpolation Mapping:** Define the exact templating engine and the required interpolation variables (`{context}`, `{query}`, `{memory_slice}`) for every system prompt before execution.
2.  **Mandate Negative Constraints:** Inject mandatory `<Safety_Rules>` and explicit destructive-action boundaries into all root prompts. For strict output enforcement, explicitly supply the JSON schema within the prompt, e.g.: "You must respond strictly in JSON format matching this exact schema: `{"action": string, "parameters": object}`".
3.  **Input Sanitization Layer:** Build a rigid serialization boundary between the state databases and the prompt injection templates to mitigate persistent, second-order prompt injection attacks originating from corrupted user data like `user_profile.json`.

### Phase 5: The Risk Evaluator & Governance
1.  **Default-Deny Authorization:** Rebuild the user authentication models to default to least-privilege (`is_admin=0`). 
2.  **Sandboxed Execution:** Ensure the `RiskEvaluator` intercepts all CLI, Web, and Automation payloads, strictly verifying permissions and confirming the intent schema before allowing the `LLMOrchestrator` to dispatch tools.
