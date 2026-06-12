# 07. Project Structure

## 1. Executive Intent (WHY this subsystem exists)
The physical project structure of Jarvis is not merely an organizational convenience; it is a rigid structural contract that physically isolates system capabilities into bounded contexts. It exists to enforce dependency injection rules, prevent circular dependencies between autonomous components, and isolate ephemeral runtime states from persistent memory stores and static business logic. This layout allows isolated execution, testing, and evolution of the LLM orchestration layer without breaking hardware interfaces or integration pipelines.

## 2. Core Domains & Responsibilities (WHAT responsibility each owns)

### `/config` - The Environmental Invariants
**Responsibility:** Owns system-wide metadata, environment variables, and bootstrap configurations (`jarvis.ini`, `settings.env`, `ai_os.json`).
**Contract:** Defines the static rules of engagement before runtime begins. It is the single source of truth for secrets and operational flags.

### `/core` - The Functional Brain
**Responsibility:** Houses the vast majority of Jarvis's deterministic business logic, state machines, and autonomous reasoning loops. 
It is strictly subdivided into discrete functional domains:
- **`agent` & `agentic`**: Owns the sub-loops, mission management, and execution traces.
- **`automation` & `tools`**: Owns deterministic local actions (UI interaction, desktop control, web tools).
- **`autonomy` & `planner`**: Owns risk evaluation, long-term scheduling, and DAG-based task orchestration.
- **`llm`**: Owns all prompt formatting, LLM routing (cloud vs. local/Ollama), and telemetry.
- **`memory`**: Owns the semantic context compression, code indexing, and RAG retrieval pipelines.
- **`voice`**: Owns the audio subsystem, spanning wake words, TTS (ONNX), and STT (Whisper) loops.
- **`runtime` (internal to core)**: Owns the application bootstrap, dependency container, and event bus.

### `/integrations` - The External Bridge Layer
**Responsibility:** Manages all third-party API communication. Contains the `loader.py` and `registry.py` to dynamically load isolated `clients/` (e.g., GitHub, Gmail, Home Assistant, Spotify).
**Contract:** Abstracts away all proprietary REST/RPC payloads. The core engine never speaks to an API directly; it only speaks to an integration client.

### `/data` & `/memory` - The Persistence Layers
**Responsibility:** Owns all state that must survive a system restart. 
- **Relational:** `jarvis_memory.db`, `auth.db` handling entity relations and credentials.
- **Vector:** `chroma/` holding semantic search indices.
- **Artifacts:** `voices/` and `whisper/` holding binary AI model weights.
- **Structured:** `goals.json`, `calendar.ics`, `user_profile.json`.

### `/runtime` & `/workspace` - The Ephemeral Operational Layers
**Responsibility:** The dynamic working grounds for the active agent.
- **`/runtime`**: Contains system execution outputs like `logs/`, session states, and `automation_state.json`.
- **`/workspace`**: Acts as a sandbox and local dropbox (`jarvis_dropbox/`) where the system reads raw commands, stages outputs (e.g., screenshots for vision tasks), and processes RAG documents.

### `/dashboard` & `/workflows`
**Responsibility:** `/dashboard` provides an external web UI for telemetry and manual overrides. `/workflows` holds static structural blueprints for automated pipelines.

## 3. Structural Interfaces (HOW it interacts)
1. **Bootstrap Sequence:** `core/runtime/bootstrap.py` reads `/config`, initializes the dependency container, and fires up the event bus.
2. **Dynamic Loading:** The `registry` subsystems in `/core` and `/integrations` scan their respective directories and register tools and clients dynamically.
3. **Execution Flow:** The LLM orchestrator in `/core/llm` generates intents, which are routed by `/core/controller` to the appropriate module in `/core/tools`, `/core/voice`, or `/integrations/clients`.
4. **State Mutations:** All state changes triggered by tools or clients are committed via `/core/memory` into the physical files in the `/data` and `/memory` directories.
5. **Asynchronous Hand-off:** Long-running inputs (like a user file drop) arrive in `/workspace/jarvis_dropbox`. The `/core/proactive` monitor detects it, triggering an event bus notification.

## 4. Systemic Fragility (WHAT breaks if removed)
- **Removing `/core`:** The entire system collapses into an inert file server. No routing, no logic, no LLM bridging.
- **Removing `/config`:** The bootstrap sequence instantly aborts. The application cannot bind to ports, authenticate with LLMs, or locate its memory stores.
- **Removing `/data` or `/memory`:** The agent becomes totally amnesiac. It loses all tool state, RAG capabilities, long-term goals, user profile context, and local TTS/STT weights, degrading it to a factory-reset state that cannot even speak.
- **Removing `/integrations`:** The agent is localized. It can reason and perform desktop actions but is completely severed from the external internet ecosystem (no email, no GitHub, no smart home control).
- **Removing `/workspace`:** Asynchronous task ingestion fails. RAG drops break, and vision capabilities fail because the agent has nowhere to stage intermediate screenshots or command payloads.

## 5. Clean Room Reconstruction (HOW to rebuild from scratch)
If rebuilding without source code, the project structure MUST be defined top-down based on capability boundaries:
1. **Define the Base Layout:** Create isolated physical boundaries: `/config`, `/core`, `/integrations`, `/data`, `/workspace`.
2. **Abstract the Core:** Inside `/core`, create strict directory partitions for each cognitive subsystem: `/llm` (reasoning), `/memory` (recall), `/planner` (DAG execution), `/voice` (audio I/O), and `/tools` (local actuators). Enforce a rule that cross-communication only happens via an `/core/runtime/event_bus.py`.
3. **Implement the Interfaces:** Define a base abstract class in `/integrations/base.py` that all third-party APIs must inherit from. Do the same for tools in `/core/tools/`.
4. **Establish the Sandbox:** Create `/workspace` as an ephemeral scratchpad where file-system events trigger system interrupts. Apply strict OS-level chroot or dockerized boundaries to prevent path traversal.
5. **Segregate Memory:** Route all long-term writing to `/data` using a unified SQLite/Vector store adapter to prevent logic modules from handling explicit file I/O. Eliminate the "split-brain paradox" by ensuring only ONE master database schema (`jarvis_memory.db`) is initialized, explicitly deprecating any legacy `memory.db` duplicates.

## 6. Structural Boundaries & Security Enforcement
To resolve vulnerabilities related to unbounded storage and sandbox escape, the following structural constraints must be physically enforced:
* **Symlink & Traversal Protection:** All read/write operations within `/workspace` must be wrapped in a realpath resolver that strictly asserts the target file resides within the absolute path of `/workspace`. Any `../` traversal attempt must trigger a fatal security halt.
* **Storage Quotas & Rotation:**
  * `/runtime` (e.g., `automation_state.json`): Must enforce a rolling limit (e.g., max 10,000 entries) and rotate older entries to prevent JSON unbounded array OOM crashes.
  * `/workspace/jarvis_dropbox/`: Must implement a maximum directory size constraint (e.g., 500MB) to prevent disk-exhaustion DoS attacks.
* **Unified State Store:** Eliminate database schema drift. All subsystems MUST refer to the absolute path of `jarvis_memory.db` provided by the central Configuration subsystem, avoiding hardcoded relative paths that accidentally spawn duplicate `.db` files.

## 7. Literal Programmatic Schemas

### Project Layout Schema (JSON Representation)
```json
{
  "project_root": "Jarvis/",
  "directories": {
    "config/": ["jarvis.ini", "settings.env", "ai_os.json"],
    "core/": ["agent/", "automation/", "autonomy/", "llm/", "memory/", "voice/", "runtime/"],
    "integrations/": ["clients/", "registry.py", "loader.py"],
    "data/": ["jarvis_memory.db", "auth.db", "chroma/", "voices/"],
    "runtime/": ["logs/", "automation_state.json", "control_flags.json"],
    "workspace/": ["jarvis_dropbox/"]
  }
}
```

### Storage Quota Configuration Schema
```json
{
  "storage_limits": {
    "workspace_max_mb": 500,
    "runtime_json_max_entries": 10000,
    "db_backup_retention_days": 7
  },
  "security": {
    "enforce_realpath_jail": true,
    "allow_symlinks_in_workspace": false
  }
}
```
