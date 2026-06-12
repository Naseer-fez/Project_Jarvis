# Architecture Overview

The Jarvis system follows a Modular Monolith architecture orchestrated by a central execution engine, separated into highly decoupled internal domains. The goal of this architecture is to provide an autonomous, multi-modal LLM-driven agent framework with extensible external integrations.

## Core Architectural Patterns

1. **Dependency Inversion via Registries**: The core system (`core`) never directly imports `integrations`. Instead, plugins register their schemas with the `CapabilityRegistry` during the `root` bootstrap process. This ensures `core` remains completely agnostic of the external services it manipulates.
2. **Actor / Loop Driven Execution**: The system revolves around the `AgentLoopEngine` which maintains context, dispatches sub-tasks via `TaskPlanner`, evaluates risk boundaries, and delegates actual function execution to capability providers.
3. **Event-Driven Inter-Process Communication**: `core.runtime.event_bus` serves as the nervous system, allowing UI (`dashboard`), Background monitors, and autonomous routines to interact entirely via pub/sub messaging.
4. **Agentic Memory Architecture**: Relies on a `HybridMemory` approach combining relational databases (`sqlite3`) for rigid schemas (profiles, sessions, logs) and vector databases (`chromadb`) for semantic recall and code embeddings.

## Key Subsystems
- **Autonomy & Governance**: `AutonomyGovernor`, `RiskEvaluator`, `PermissionMatrix`. Ensures actions are safe to execute and requests user confirmation for destructive operations.
- **LLM Routing**: `ModelRouter`, `LLMClientV2`. Manages failovers, fallback loops, and cost optimization between various cloud and local (`Ollama`) models.
- **Tools & Desktop Observation**: Uses `VisionTool`, `DesktopActionExecutor`, and `DesktopObserver` to dynamically interact with the host OS environment.