# Runtime Investigator Report: engine.py

## Role Relevancy
The core asynchronous DAG execution engine that executes step plans defined by the agent.

## Assumptions
- Uses topological sorting to respect graph dependencies.
- Features LIFO reverse-topological rollback for steps defining `rollback` schemas upon failure.
- Halts execution and fails the task entirely if a single step permanently fails.
- Injects replay semantics reading `_step_results` and `_replay_active` from the context variables to skip successful steps.
- Captures pre- and post-step state snapshots via `context.save_snapshot`.
- Retries step failures with exponential backoff (`backoff *= 2.0`) up to `retry_count`.

## Schema & API Contracts
- Accepts `TaskExecutionContext` state containers.
- Execution steps: dicts containing `id`, `action` / `tool`, `params`, `retry_count`, `rollback`, `depends_on`.
- Returns dict: `{"status": "success"|"failure", "results": {...}, "error": str}`.

## Dependencies
- `core.context.context.TaskExecutionContext`
- `core.executor.dag.PlanDAG`
- Integration with ToolRouter/CapabilityRegistry and `AutonomyGovernor`.

## Configuration Variables
- None explicit, configurable `retry_count` injected via step definitions.

## Prompts
- None.
