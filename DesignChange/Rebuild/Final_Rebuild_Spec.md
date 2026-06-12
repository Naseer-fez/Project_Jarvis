# Final Rebuild Specification

This document serves as the master specification for completely rebuilding the autonomous Jarvis assistant.

## Architectural Constraints
- **Pattern**: Modular Monolith.
- **Dependency Inversion**: Core must not directly import integration dependencies. Tools must be injected via a capability registry.
- **Language & Runtime**: Python 3.11+, heavily relying on `asyncio`.
- **Database**: `aiosqlite` for state management; `chromadb` for vector RAG memory.

## Subsystem Specifications

### 1. `core`
The brain of the operation.
- Must implement an `AgentLoopEngine` that iterates through: Observation -> Reflection -> Goal Formulation -> Planning -> Tool Execution -> Memory Update.
- Must implement `AutonomyGovernor` to halt execution on `HIGH` risk tool invocations (e.g., destructive actions or email sending) until a user approves.
- Must implement `ModelRouter` to route tasks to either Local LLMs (`Ollama`) for simple queries, or Cloud LLMs (`GPT-4` / `Claude-3`) for complex logic.

### 2. `integrations`
The hands of the operation.
- Must implement `BaseIntegration` schema exposing `is_available()` and `get_tools()`.
- Must encapsulate API calls (e.g., `github`, `gmail`, `spotify`, `telegram`, `weather`).
- External API limits and network timeouts must be gracefully caught and returned to the `core` as `ToolObservation` errors.

### 3. `dashboard`
The face of the operation.
- Must implement a `FastAPI` server.
- Must provide endpoints to start/stop the core daemon.
- Must provide a WebSocket to stream live `JarvisState` updates to the frontend.

### 4. `audit`
- Must log every single state transition, LLM request/response payload, and tool execution using `logging.handlers.QueueHandler` to avoid blocking main threads.