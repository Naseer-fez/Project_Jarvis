# Data Flow Architecture

## 1. System Intent (WHY does this subsystem exist?)
The Data Flow Subsystem forms the central nervous system of Jarvis, existing to bridge the gap between ephemeral real-time interactions and persistent, long-term semantic understanding. Without it, Jarvis would be an amnesiac input-output function incapable of statefulness. It exists to provide a reactive execution lifecycle, allowing the agent to continuously learn preferences, remember factual milestones, manage context windows efficiently, and operate across multiple decoupled components without blocking main thread execution. 

## 2. Core Responsibilities (WHAT responsibility does it own?)
- **State Governance**: Enforcing a strict, central thread-safe `StateMachine` (`AgentState`) that dictates what the agent is currently doing (e.g., `IDLE`, `THINKING`, `PLANNING`, `EXECUTING`) and safeguarding against invalid behavioral transitions.
- **Event Broadcasting**: Operating a publish/subscribe `EventBus` that decouples modules, permitting disparate systems (UI, metrics, logging) to react asynchronously to state changes, payloads, and occurrences.
- **Hybrid Memory Management**: Synchronizing two distinct storage paradigms:
  - Exact relational storage (SQLite via `SQLitePool` & `SQLiteStorage`) for deterministic audits, tool actions, and raw conversation logs.
  - Semantic vector storage (ChromaDB via `SemanticMemory`) for fuzzy, intent-based retrieval and RAG capabilities.
- **Profile Synthesis**: Continuously extracting identity constraints and user preferences via the `UserProfileEngine` to maintain an evolving `user_profile.json`, tracking interaction counts, language traits, and stylistic tendencies.
- **Context Compression**: Dynamically shrinking and deduplicating multi-modal memory recall results via a `ContextCompressor` into token-efficient context blocks before injection into the LLM prompt space.

## 3. System Interactions (HOW does it interact with the rest of the system?)
- **Agent Loop Injection**: As the `AgentLoopEngine` processes a goal, it queries the `MemoryRetriever` for relevant past context. The Data Flow system fetches data synchronously from both SQLite and ChromaDB, hands it to the `ContextCompressor`, and injects the resulting compressed context block into the LLM's system prompt alongside parameters from the `UserProfileEngine`.
- **Decoupled Reactions**: The `StateMachine` emits transition events to the `EventBus`. Subsystems register observers (via callbacks) to listen for these events, reacting without needing direct dependencies on the Agent's core logic.
- **Background Synthesis**: The `MemorySubsystem` acts as a reactive daemon. When conversation buffers overflow or specific heuristic rules trigger, it dispatches non-blocking async tasks to synthesize raw dialogue into permanent "Episodes" or preference deltas. It then applies them back to the semantic database and `user_profile.json` using atomic file writes to prevent state corruption.
- **Execution Audit**: Every tool execution, confirmation block, and state change is durably recorded into SQLite, leaving an immutable breadcrumb trail (`ExecutionTrace`) for the `RiskEvaluator` or human operators to audit.

## 4. Failure Modes (WHAT would break if removed?)
- **Amnesia**: Removing Hybrid Memory would prevent Jarvis from recalling past conversations, learned facts, or codebase indexing, reducing the agent to a zero-shot, contextless entity completely incapable of personalization.
- **State Corruption & Deadlocks**: Removing the State Machine would lead to fatal race conditions where multiple subagents attempt to speak, listen, or execute tools simultaneously, resulting in port conflicts, audio crashing, and corrupted execution states.
- **Coupling Gridlock**: Removing the Event Bus would require hardcoding monolithic connections between the Agent Loop, UI, and telemetry subsystems, causing catastrophic circular dependency chains and breaking all background async operations.
- **Token Exhaustion**: Removing the Context Compressor would flood the LLM context window with unstructured, redundant recall data, leading to massive API latency, runaway token costs, and hallucinated associations.

## 5. Reconstruction Strategy (HOW would it be rebuilt from scratch without source code?)
To recreate the Data Flow Subsystem from scratch without access to the original source:
1. **Design the Core State Machine**: Implement a thread-safe singleton Enum state manager with a predefined transition matrix governing allowed state paths. Incorporate context managers (`StateGuard`) for temporary state overrides that automatically revert upon block exit.
2. **Implement an Asynchronous Event Bus**: Create an in-memory Pub/Sub message broker capable of handling fire-and-forget events, backed by a bounded replay history deque to allow late-joining subscribers to catch up on recent activity.
3. **Build the Dual-Path Memory Controller**:
   - Establish a SQLite database using WAL (Write-Ahead Logging) mode to handle concurrent reads/writes for conversation histories, actions, and key-value preference facts.
   - Establish an embedded vector database to store and query text embeddings of conversation turns and milestone episodes using cosine similarity.
4. **Develop the Context Compressor**: Write an algorithm that takes the top-K results from both vector search and SQLite exact-match queries, deduplicates them, applies time-decay weighting, truncates based on predefined token limits, and formats the output into a dense context block.
5. **Implement Background Consolidation**: Create a background event listener that periodically runs an LLM summarization pipeline over the raw conversation history. This pipeline should extract distinct "Episodes" and "Preferences", updating a core JSON profile file via atomic tmp-file replacement while simultaneously injecting new vectors into the semantic database.

## 6. Structural Contracts and Data Schemas

To address split-brain fragmentation, timestamp drift, and missing structural contracts, the following literal schemas MUST be implemented exactly as specified:

### 6.1 EventBus Contract (`EventRecord`)
The `EventBus` operates on this exact dataclass to guarantee event structure across the system:
```python
class EventRecord:
    event_id: str
    event_type: str
    payload: Any
    source: str
    created_at: float
```

### 6.2 SQLite Storage: `jarvis_memory.db` Schema
This database manages exact relational storage. Implement the following schemas:

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
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    event TEXT,
    category TEXT,
    timestamp TEXT
);

CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_input TEXT,
    assistant_response TEXT,
    session_id TEXT,
    timestamp TEXT
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

### 6.3 SQLite Storage: `memory.db` Schema
Historical / overlapping schemas for persistence and configuration. Notice the timestamp and typing variations that must be unified:

```sql
CREATE TABLE preferences (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE episodic_memory (
    id INTEGER,
    event TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category TEXT
);

CREATE TABLE conversation_history (
    id INTEGER,
    user_input TEXT,
    assistant_response TEXT,
    timestamp TIMESTAMP,
    session_id TEXT
);

CREATE TABLE episodes (
    id INTEGER,
    content TEXT,
    category TEXT,
    created_at TEXT,
    timestamp TEXT DEFAULT ''
);

CREATE TABLE conversations (
    id INTEGER,
    user_input TEXT,
    assistant_response TEXT,
    session_id TEXT,
    timestamp TEXT
);

CREATE TABLE actions (
    id INTEGER,
    action TEXT,
    result TEXT,
    success INTEGER DEFAULT 1,
    metadata TEXT,
    timestamp TEXT
);
```

### 6.4 Profile Profile Definition (`user_profile.json`)
The structured DTO mapping for `user_profile.json`. To prevent implicit god-mode defaults, inputs mapping to this object must be strictly sanitized.

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
