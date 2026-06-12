# Runtime Investigator Report: agent_loop.py

## Role Relevancy
Core implementation of the primary agent execution loop: plan -> risk -> confirm -> execute -> reflect.

## Assumptions
- Uses an asynchronous workflow for execution (`DAGExecutor`).
- Task-level execution is hardcoded to a 5-minute timeout (`asyncio.timeout(300)`).
- `<think>` tags (common in deepseek-r1 outputs) are regex-extracted and cleaned.
- Headless environments or `LEVEL_4` autonomy bypass manual confirmation steps.
- Uses `TaskExecutionContext` state machine for transitioning execution phases.

## Schema & API Contracts
- **ExecutionTrace**: Data class tracking `goal`, `iterations`, `plan`, `observations`, `risk_scores`, `think_blocks`, `reflection`, `final_response`, `success`, `stop_reason`, and timestamps.
- **DAG Plan Structure**: Dictionary containing a `steps` list. Each step has `id`, `action`, `description`, `params`.
- **Observations**: `ToolObservation` results mapped back from executor.

## Dependencies
- `core.state_machine.State`, `StateMachine`
- `core.context.context.TaskExecutionContext`
- `core.autonomy.autonomy_governor.AutonomyGovernor`
- `core.autonomy.risk_evaluator.RiskEvaluator`
- `core.planner.planner.TaskPlanner`
- `core.metrics.confidence.ConfidenceModel`
- `core.registry.registry.CapabilityRegistry`

## Configuration Variables
- `_DEFAULT_MAX_ITERATIONS = 10`
- `model = "deepseek-r1:8b"`
- `ollama_url = "http://localhost:11434"`

## Prompts
- Found `REFLECT_SYSTEM_PROMPT`. Extracted to prompts directory.
