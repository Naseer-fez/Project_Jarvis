# Runtime Investigator Report: dispatcher.py

## Role Relevancy
Wrapper around `DAGExecutor` enforcing execution pathway bounding limits.

## Assumptions
- Enforces a hardcoded max recursion depth to prevent unbounded agent loop runaway conditions.
- Raises `RecursionError` on threshold breach.

## Schema & API Contracts
- `DispatchPipeline`: `execute(plan, context, current_depth)`.

## Dependencies
- `core.executor.engine.DAGExecutor`

## Configuration Variables
- `max_recursion_depth = 5`

## Prompts
- None.
