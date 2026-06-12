# Runtime Investigator Report: context.py

## Role Relevancy
Provides isolated execution scoping, correlation IDs, tracing semantics, and state machine encapsulation for tasks.

## Assumptions
- Exposes async/sync context management `__enter__` utilizing `contextvars.Token` for `trace_id` injection into logs.
- Automatic exception catching in exit scope translates to error transition on `StateMachine`.
- Serializes `save_snapshot` to the `logs/traces` local folder.

## Schema & API Contracts
- `TaskExecutionContext` properties: `trace_id`, `task_id`, `variables` (kv store), `logs` list.
- API: `.log()`, `.get()`, `.set()`, `.to_dict()`, `.save_snapshot()`.

## Dependencies
- `core.state_machine.StateMachine`
- `core.logging.logger.set_trace_ids`

## Configuration Variables
- Defaults: snapshot dir `logs/traces/{trace_id}.json`.

## Prompts
- None.
