# Cross-Domain Flows

Based on the actual structural analysis of the codebase, the interactions across the major domain boundaries (`core`, `dashboard`, `integrations`, `audit`, and `root`) form a strictly orchestrated execution graph centered heavily around the `core` domain.

## Primary Interaction Interfaces

### 1. `dashboard` -> `core` (User Request Ingestion)
- **Ingestion**: The FastApi endpoints in `dashboard\server.py` accept external inputs (`CommandRequest`, `GoalAddRequest`).
- **Delegation**: The requests are forwarded to `core.controller_v2.JarvisControllerV2` or `core.controller.goal_runner.GoalRunner`.
- **Feedback**: The dashboard polls or subscribes to events emitted by the `core.runtime.event_bus` or `core.state_machine` to return `CommandResponse` or stream updates back to the UI.

### 2. `core` -> `integrations` (Tool Execution)
- **Registry Lookup**: When `TaskPlanner` or `LLMOrchestrator` decides an action requires an external tool, it invokes `core.registry.CapabilityRegistry`.
- **Execution Hand-off**: Tools registered via `integrations.registry.IntegrationRegistry` are executed. Examples include `EmailIntegration._send_email` or `GitHubIntegration._create_issue`.
- **Response Mapping**: Outputs from the integrations are wrapped into `core.capability.base.ToolObservation` objects and fed back into the `AgentLoopEngine` context.

### 3. `core` -> `audit` (Telemetry & Verification)
- **Logging**: All internal states, LLM call outputs, and tool observations are logged into `audit.audit_logger`.
- **Traceability**: The `core.agent.agent_loop.ExecutionTrace` ensures that steps taken during an autonomous mission are safely committed to the ledger.

### 4. `root` -> `core` / `dashboard` (Bootstrapping)
- Scripts like `main.py` handle environment initialization via `core.runtime.bootstrap`, mount the `core.controller.services`, start the FastAPI `dashboard`, and attach signals for safe shutdown.

## Advanced Asynchronous Flows
- **Proactive Execution**: `core.proactive.background_monitor.BackgroundMonitor` and `core.automation.live_automation.LiveAutomationEngine` continually scan the filesystem or incoming events independent of user input, creating goals inside `core.autonomy.goal_manager.GoalManager` which then trigger the standard `core` -> `integrations` workflow.