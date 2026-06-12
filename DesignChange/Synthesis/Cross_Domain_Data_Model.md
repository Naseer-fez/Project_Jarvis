# Cross-Domain Data Model

The data structures in Jarvis are defined within `core` but traverse all outer domains (`dashboard`, `integrations`, and `audit`).

## Shared Entities

### 1. The Observation & Result Schema
Data returned by `integrations` and tools.
- **`ToolObservation` / `ToolResult`** (`core.capability.base`): The common interchange format. Used by `LLMOrchestrator` to feed context back into LLMs.
- **`DesktopObservation` / `DesktopActionResult`** (`core.desktop.contracts`): Specialized payloads when vision or GUI tools are utilized.

### 2. Goals and Execution Context
- **`Goal` / `GoalStatus`** (`core.autonomy.goal_manager`): Tracks high-level objectives. `dashboard` reads this for UI updates.
- **`TaskExecutionContext`** (`core.context.context`): Carries state through the `DAGExecutor` and `AgentLoopEngine`.
- **`ExecutionTrace` / `MissionStepRecord`**: Immutable records of what occurred. Synced directly into `audit` and `core.logging.logger`.

### 3. State Machine Signatures
- **`JarvisState` / `State`** (`core.state_machine`): Global states like `Idle`, `Thinking`, `Executing`, `Waiting`. Emitted by `core`, monitored by `dashboard.server.JarvisState`.

### 4. Memory Constructs
- **`_Fact` / `ToolObservation`**: The `core.memory.hybrid_memory.HybridMemory` ingests context derived from `integrations` and user interactions, storing it both in `SQLiteStorage` and embedding it via `EmbeddingManager`.

## Persistence Models
- **Relational (`aiosqlite` / `sqlite3`)**: User profiles, historical logs, permission matrices.
- **Vector (`chromadb`)**: Semantic memory of facts and codebase embeddings (`CodeIndexerService`).
- **File System**: Temporary files for automation and image payloads (`core.automation.payload_extractor`).