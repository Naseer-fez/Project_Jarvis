# Analysis Report for context.py

## Dependencies
- __future__.annotations
- logging
- uuid
- typing.Any
- contextvars.Token
- core.state_machine.StateMachine
- core.state_machine.State
- core.logging.logger.set_trace_ids
- core.logging.logger.reset_trace_ids

## Schemas
- TaskExecutionContext

## API Contracts
- TaskExecutionContext.__init__(self, trace_id, task_id, event_bus, state_machine)
- TaskExecutionContext.log(self, message, level)
- TaskExecutionContext.get(self, key, default)
- TaskExecutionContext.set(self, key, value)
- TaskExecutionContext.__getitem__(self, key)
- TaskExecutionContext.__setitem__(self, key, value)
- TaskExecutionContext.__contains__(self, key)
- TaskExecutionContext.to_dict(self)
- TaskExecutionContext.__enter__(self)
- TaskExecutionContext.__exit__(self, exc_type, exc_val, exc_tb)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: TaskExecutionContext — holds correlation IDs, state machine, execution logs,
and variables isolated to a single task execution flow.

