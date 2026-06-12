# Data Models & System Semantics

This document defines the core data topologies, persistence contracts, and deep semantic models driving Jarvis. It operates under the premise that data structures dictate system intelligence: models are not just storage containers, they are the functional boundaries defining what the AI can perceive, remember, and safely execute.

---

## 1. User Identity & Persona Model (`user_profile.json`)

**WHY does this subsystem exist?**
To anchor the LLM's dynamic alignment. A stateless LLM resets to a generic persona every session. This model exists to construct a persistent "memory of the user," enabling the AI to modulate its tone, adapt to technical competence, and track engagement lifecycle.

**WHAT responsibility does it own?**
It acts as the system's internal psychological and demographic profile of the primary operator. It tracks immutable preferences (timezone, language) and mutable analytical metrics (interaction counts, behavioral timestamps like `first_seen`/`last_seen`, inferred `expertise_level`, and `communication_style`).

**HOW does it interact with the rest of the system?**
During the context-assembly phase of the agent loop, this model is injected into the foundational System Prompt. It interacts with the session manager, which hooks into user activity to perform thread-safe atomic writes, continuously updating engagement metrics. 

**WHAT would break if removed?**
Jarvis would suffer total persona amnesia. All responses would revert to default, unaligned base-model behaviors. Contextual assumptions about user location, timezone-dependent scheduling, and preferred verbosity would completely fail.

**HOW would it be rebuilt from scratch?**
Design a lightweight Root DTO (Data Transfer Object) tracking strings for localization and preferred topics, backed by JSON persistence. Implement a middleware hook on all user input events to update a `last_seen` ISO-8601 timestamp and increment an `interaction_count` integer, writing synchronously to disk.

**Exact Programmatic Schema:**
```json
{
  "name": "string",
  "communication_style": "string",
  "expertise_level": "string",
  "preferred_topics": ["string"],
  "timezone": "string",
  "language": "string",
  "interaction_count": 0,
  "first_seen": "YYYY-MM-DDTHH:MM:SS.mmmmmm",
  "last_seen": "YYYY-MM-DDTHH:MM:SS.mmmmmm"
}
```

---

## 2. Deterministic Relational Memory (`memory.db` / `jarvis_memory.db`)

**WHY does this subsystem exist?**
To guarantee absolute, transactional recall. While vector databases provide fuzzy relevance, Jarvis requires an immutable source of truth for explicit configuration overrides, historical conversational exact-matches, and audit logs of its physical actions.

**WHAT responsibility does it own?**
Managing the multi-table SQLite datastore operating in WAL (Write-Ahead Logging) mode. It owns the rigid topologies for:
- **Preferences**: A strict Key-Value registry for system and user configuration flags.
- **Episodic Memory**: Distinct factual events and lifecycle milestones, tagged by category.
- **Conversation History**: Precise mapping of User Input to Agent Response mapped via `session_id`.
- **Action Audit Logs**: A rigorous ledger of tool executions, capturing boolean success states, input metadata payloads, and output results.

**HOW does it interact with the rest of the system?**
It is the backbone of the `memory_subsystem`. The `context_compressor` queries it to reconstruct recent chronological states. Tool execution wrappers commit to the `actions` table synchronously to ensure the AI knows exactly what it just did before planning its next move.

**WHAT would break if removed?**
The AI would lose the ability to recall explicit instructions or configurations ("always use direct communication"). It would have no deterministic record of which tools succeeded or failed in the past, leading to infinite loops of repeating failed actions. 

**HOW would it be rebuilt from scratch?**
Deploy a relational SQLite database with strict `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` schemas. Build indexing on timestamps for fast descending chronological retrieval. Enforce concurrent access safety via WAL mode, and map the schema to DTOs for Preferences, Episodes, Turns, and Actions.

**Exact Programmatic Schema:**
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

CREATE INDEX IF NOT EXISTS idx_preferences_updated_at ON preferences(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp DESC);
```

---

## 3. Associative Semantic Memory (`chroma.sqlite3` & Vector DB)

**WHY does this subsystem exist?**
To simulate human intuitive recall. It exists to map ambiguous, fuzzy natural language queries against the mathematical embedding space of all prior historical context.

**WHAT responsibility does it own?**
Translating high-dimensional embedding vectors into relevant context chunks. It abstracts the tenant, segment, and scope metadata, matching historical episodes, code snippets, and conversational turns against the user's immediate intent.

**HOW does it interact with the rest of the system?**
Tightly coupled to the embedding provider and the `retriever` pipeline. Before an LLM orchestration call, the planner module queries the semantic memory to extract the Top-K most relevant contextual artifacts to inject into the prompt context window.

**WHAT would break if removed?**
Jarvis would lose contextual awareness of analogous past situations. RAG (Retrieval-Augmented Generation) would collapse, forcing the system to rely entirely on immediate short-term conversation history and static base knowledge.

**HOW would it be rebuilt from scratch?**
Implement a vector storage engine (like ChromaDB or FAISS). Define logical collections separating `episodes`, `preferences`, and `conversations`. Write an ingestion pipeline that chunks text, computes float-array embeddings via an external model, and stores them with UUIDs and keyword-searchable metadata.

**Exact Programmatic Schema & Index Parameters:**
```python
# ChromaDB Index Parameters
metadata = {"hnsw:space": "cosine"}

# Required Collections
collections = [
    "jarvis_preferences",
    "jarvis_episodes",
    "jarvis_conversations"
]
```

---

## 4. Desktop Automation Domain Models (`desktop/contracts.py`)

**WHY does this subsystem exist?**
To bridge the semantic gap between textual LLM intent and brittle, coordinate-based physical GUI manipulation. It acts as a Domain-Driven Design (DDD) layer to enforce strict typing on actions that mutate the host OS.

**WHAT responsibility does it own?**
Standardizing the visual and executable boundaries of the OS. 
- `DesktopObservation`: Defines what the AI "sees" (OCR text, screenshot fingerprints, confidence scores, target bounding boxes).
- `DesktopAction`: Defines what the AI "does" (click, type, hotkey), strictly mapping parameters, expected outcomes, and risk tiers.
- `DesktopActionResult`: Defines the outcome, tracking execution duration, success status, and error traces.

**HOW does it interact with the rest of the system?**
The LLM generates an intent mapped to a `DesktopAction`. This payload passes through the `risk_evaluator` (which assesses `DesktopRiskTier` for manual approval requirements) before hitting the `auto_clicker` or execution engine. The result is captured as a `DesktopObservation` and returned to the State Machine.

**WHAT would break if removed?**
The GUI automation pipeline would devolve into dangerous, untyped hallucinated tool calls. The risk evaluation engine would have no structured data to assess, bypassing safety protocols, and vision models would lack standardized bounding boxes to target.

**HOW would it be rebuilt from scratch?**
Define abstract dataclasses for Screen Targets (x, y, width, height, confidence). Enforce Enum restrictions on action types (e.g., `CLICK`, `TYPE_TEXT`). Require every physical action to define an `expected_change` string and a `risk_tier` before the payload is considered valid for execution.

---

## 5. Idempotency & Proactive Goal State (`automation_state.json` & `goals.json`)

**WHY does this subsystem exist?**
To decouple long-running autonomous workflows from the volatile, ephemeral memory of the active conversational agent thread.

**WHAT responsibility does it own?**
- `automation_state.json`: Maintains a deduplication ledger (`seen_fingerprints`) of cryptographic hashes for processed files, ensuring idempotent ingest operations.
- `goals.json`: Tracks the queued proactive missions, cron-like scheduled tasks, and autonomous intents that persist across system reboots.

**HOW does it interact with the rest of the system?**
The background monitor and automation manager poll these lightweight JSON stores upon bootstrap. The ingestion pipeline checks fingerprints before processing documents. The scheduler reads goals to trigger proactive agent loops without user intervention.

**WHAT would break if removed?**
The system would repeatedly process the same documents in infinite ingestion loops. Autonomous, scheduled tasks would vanish if the application restarted, destroying Jarvis's ability to act proactively.

**HOW would it be rebuilt from scratch?**
Implement a file-backed JSON serialization layer. For idempotency, maintain an array of SHA-256 string hashes that is updated and flushed to disk after every successful file ingest. For goals, build an array of task DTOs containing target execution datetimes and action payloads.

**Exact Programmatic Schemas:**
`automation_state.json`:
```json
{
  "saved_at": "YYYY-MM-DDTHH:MM:SS.mmmmmm+00:00",
  "seen_fingerprints": [
    "string (SHA-256 hash)"
  ]
}
```

`goals.json`:
```json
{
  "saved_at": "YYYY-MM-DDTHH:MM:SS.mmmmmm+00:00",
  "goals": [],
  "schedule": []
}
```

---

## 6. Macro-Cognitive State Machine (`core/state_machine.py`)

**WHY does this subsystem exist?**
To impose deterministic physics on a non-deterministic generative system. It governs the lifecycle of the AI's internal thought process.

**WHAT responsibility does it own?**
Tracking and restricting the active mode of the agent loop via an Enum (`IDLE`, `THINKING`, `PLANNING`, `EXECUTING`, etc.). It acts as a strict Finite-State Machine, preventing illegal architectural transitions (e.g., executing an action before risk evaluation). 

**HOW does it interact with the rest of the system?**
It is universally referenced by the controller and execution engines. UI dashboards and telemetry services subscribe to state transition events to visualize what the AI is currently doing. Thread-safe `StateGuard` context managers lock the state during sensitive operations.

**WHAT would break if removed?**
The execution loop would suffer from race conditions. The AI might attempt to speak, listen, and execute code simultaneously. Dashboards would fail to reflect current operations, rendering the system opaque to the user.

**HOW would it be rebuilt from scratch?**
Define an Enum of valid system states. Construct a Directed Acyclic Graph (DAG) mapping strictly allowed transitions. Implement a thread-safe observer pattern to notify subscribed telemetry channels whenever a state transition successfully occurs, rejecting any mutations not defined in the graph.
