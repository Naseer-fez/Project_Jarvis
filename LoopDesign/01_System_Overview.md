# 01 System Overview: Jarvis AI Operating System Framework

## 1. System Intent & Existence (WHY)
The Jarvis framework exists to eliminate the gap between abstract Large Language Model (LLM) reasoning and concrete, persistent, local-system execution. Standard LLMs are stateless and sandboxed; Jarvis exists to provide a persistent, multi-modal "operating system" layer that wraps the LLMs. It converts passive text generation into proactive, context-aware autonomous action within a Windows-based host OS. The framework exists specifically to manage the extreme complexity of asynchronous multi-step execution, ensuring that LLM decisions are translated safely into physical state changes (files, API states, synthetic desktop inputs) while maintaining an evolving relational and semantic memory of the user.

## 2. Core Responsibilities (WHAT)
The overarching Jarvis framework owns the lifecycle, state, and governance of all sub-components. Its primary responsibilities include:
*   **State & Memory Orchestration:** Unifying disparate data streams—relational data (SQLite), vector embeddings (ChromaDB), and transient runtime execution states (JSON locks/caches)—into a cohesive "context window" that limits token overflow and ensures continuity across sessions.
*   **Decoupled Control Flow Pipeline:** Operating a dual-execution model where the synchronous User Interface (CLI/Voice) remains responsive while complex, highly-latent agentic loops (AutomationManager, GoalRunner, State Machine) execute asynchronously in the background.
*   **Dynamic Model Routing:** Evaluating user intents or system events for cognitive complexity, and routing requests dynamically to specific tiered LLMs (e.g., Reflexive vs. Reasoning vs. Execution models) to optimize cost, latency, and capability.
*   **Governance & Action Sandboxing:** Strictly controlling what the LLM can modify on the host machine. It owns the permission matrix, enforcing "user-in-the-loop" confirmations via a Risk Evaluator before executing sensitive tools (CLI commands, filesystem deletion, specific API calls).
*   **Multi-Modal Abstraction:** Abstracting the complexities of environmental observation. It owns the translation layer that converts raw screen pixels and continuous audio streams into structured text/JSON payloads suitable for LLM digestion.

## 3. Interactions & Workflows (HOW)
Jarvis operates as a Facade-driven architecture backed by a central Dependency Injection (DI) Container and an asynchronous Event Bus.
*   **Internal Component Workflows:** Components do not directly instantiate each other. Instead, the DI container resolves dependencies at runtime. When the `VoiceLayer` detects a transcribed command, it does not call the LLM directly; it publishes an `IntentEvent` to the Event Bus. The `JarvisControllerV2` (Facade) subscribes to this event, packages the current state from the `MemorySubsystem`, and routes it to the `LLMOrchestrator`.
*   **External Host Interactions:** The framework interacts with the Windows OS strictly via bounded operational contracts. Filesystem manipulation is constrained to specific "workspace" directories. "Dropbox" folders are proactively monitored by background threads, triggering automatic ingestion pipelines when new data (files, images) appears.
*   **LLM Provider Workflows:** When interacting with external APIs (e.g., Ollama, Gemini), the framework heavily sanitizes and truncates inputs. It injects specific, dynamically assembled prompt architectures containing guardrails, required JSON schemas, and maximum token budgets, forcing the LLM to reply in a strict state-machine-compatible format.
*   **State Transitions:** The system transitions between `IDLE`, `LISTENING`, `THINKING`, `EXECUTING`, and `AWAITING_CONFIRMATION` states globally. Any state mutation broadcasts an event, ensuring UI components (CLI, Dashboard) remain in perfect sync with backend execution.

## 4. Failure Modes & Cascading Effects (WHAT BREAKS)
If the central framework/orchestrator is removed or bypassed, the following catastrophic failures occur:
*   **Contextual Collapse:** LLMs instantly lose the ability to maintain conversational flow or remember previous actions. Memory fragmentation occurs, leading to context poisoning and infinite generation loops.
*   **State Corruption & Race Conditions:** Without the Event Bus and DI Container, subagents attempting to write to the same relational/vector database concurrently will cause hard locks, corrupting the SQLite/ChromaDB state.
*   **Catastrophic Host Compromise:** Bypassing the Risk Evaluator and Sandboxing layers allows a hallucinating LLM direct, root-level execution paths, potentially deleting critical user files or exposing network ports without constraint.
*   **Orphaned Asynchronous Tasks:** Background tasks (e.g., automation scanning, scheduling) would become zombie threads, silently consuming RAM and CPU (OOM attacks) as they continuously poll without a central lifecycle manager to terminate them.

## 5. Reconstruction Strategy (HOW TO REBUILD)
To rebuild the Jarvis framework from scratch without access to source code, the following strict sequence must be followed to satisfy the 100% reconstruction standard:
1.  **Define the Environmental Boundaries (The Sandbox):** Establish the exact filesystem bounds (`jarvis.ini`), configuring the strict whitelist of permissible actions, directories, and timeout thresholds.
2.  **Implement the Core Primitives:** Build the central `ServiceContainer` (Dependency Injection) and `EventBus` (Pub/Sub). All future modules must strictly adhere to injecting dependencies via the container and communicating state changes solely via event propagation.
3.  **Construct the Dual-Memory Architecture:** Scaffold the SQLite schema for relational data (user profile, task history, metadata) and the ChromaDB schema for semantic search. Implement strict truncation rules and schema validation to handle data flow into these databases.
4.  **Build the Execution & Governance Engines:** Develop the `RiskEvaluator` that intersects every tool call. Construct the asynchronous State Machine and DAG Executor that manages multi-step plans, incorporating explicit rollback and timeout failure conditions.
5.  **Implement the Controller Facade (`JarvisControllerV2`):** Centralize intent routing, taking inputs from raw interfaces and dispatching them through the `LLMOrchestrator` to local or remote models. This requires meticulously reconstructing the implicit JSON/XML output schemas expected from the LLMs.
6.  **Attach the Sensory Layers (Interfaces):** Finally, construct the asynchronous CLI and Voice (STT/TTS/Wakeword) loops as "dumb" thin clients that merely push inputs to the Event Bus and render the outputs they receive, containing zero business logic.

## 6. Literal Programmatic Schemas
To satisfy the strict 100% reconstruction standard without ambiguity, the exact programmatic schemas extracted from the data models and prompt bindings are provided below.

### 6.1 Relational Memory (`memory.db` SQLite Schema)
```sql
CREATE TABLE preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_preferences_updated_at ON preferences(updated_at DESC);

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

### 6.2 Vector Search (`chroma.sqlite3` SQLite Tables)
ChromaDB internal relational mapping structure:
```sql
CREATE TABLE collections (id TEXT, name TEXT, dimension INTEGER, database_id TEXT, config_json_str TEXT, schema_str TEXT);
CREATE TABLE embeddings (id TEXT, segment_id TEXT, embedding_id TEXT, seq_id INTEGER, created_at TIMESTAMP);
CREATE TABLE segments (id TEXT, type TEXT, scope TEXT, collection TEXT);
```

### 6.3 JSON Memory & Configuration Objects

**User Profile Object (`user_profile.json`):**
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

**Automation Tracker State (`automation_state.json`):**
```json
{
  "saved_at": "2026-06-11T12:56:50.555604+00:00",
  "seen_fingerprints": [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ]
}
```

### 6.4 LLM Expected Output Interfaces (Implicit Contracts)

**GUI Control Output Interface:**
```json
{
  "found": true,
  "x": 123,
  "y": 456,
  "width": 100,
  "height": 50,
  "confidence": 0.9,
  "reason": "description of finding"
}
```

**System Bound Configuration (`jarvis.ini` section examples):**
```ini
[general]
name = Jarvis
version = 2.0.0
environment = local

[execution]
allow_app_launch = true
allow_gui_automation = true
sandboxed_execution = true
rollback_support = true

[risk]
voice_confirm_threshold = MEDIUM
failsafe_auto_disable_on_error = true
failsafe_error_threshold = 3
```
