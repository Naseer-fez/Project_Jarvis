# Module Map

The codebase is logically partitioned into 5 primary domains.

## 1. `core/` (The Monolith)
- `core.agent`: The autonomous execution loops (`AgentLoopEngine`).
- `core.autonomy`: Risk evaluation and goal tracking (`AutonomyGovernor`, `GoalManager`).
- `core.controller`: LLM dispatchers and router configurations (`JarvisControllerV2`).
- `core.desktop`: Computer control and vision implementations (`DesktopActionExecutor`).
- `core.hardware`: Serial interfaces to physical IoT hardware.
- `core.llm`: SDK clients and telemetry (`ModelRouter`, `LLMClientV2`).
- `core.memory`: RAG storage and SQL interfaces (`HybridMemory`, `CodeIndexerService`).
- `core.planner`: Subtask generation (`TaskPlanner`).
- `core.tools`: Internal built-in tools (file systems, Python shell).
- `core.voice`: Voice processing loop and handlers.

## 2. `integrations/` (The Plugins)
- `integrations.clients.*`: Specific implementations for external services (e.g., `GitHubIntegration`, `HomeAssistantIntegration`, `SpotifyIntegration`).
- `integrations.registry`: The binder for all tool schemas.

## 3. `dashboard/` (The Presentation Layer)
- `dashboard.server`: The FastAPI ASGI endpoints and WebSocket hubs.

## 4. `audit/` (The Observer)
- `audit.audit_logger`: Safe, isolated file logging for compliance and debugging.

## 5. `tests/` (The Verifier)
- Contains unit and integration test harnesses specifically mocking dependencies using dependency injection techniques (`MockToolRouter`, `MockMemory`).