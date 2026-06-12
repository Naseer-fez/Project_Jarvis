# Runtime Investigator Report: state_machine.py

## Role Relevancy
Defines the state transitions forming the backbone of the agent runtime loop execution.

## Assumptions
- Only explicitly mapped state transitions are permitted; invalid transitions raise `IllegalTransitionError`.
- Execution states maintain a rolling audit trail of 100 maximum transitions for forensics.
- Event bus integration: emits `state_transition` events.
- Re-entrant thread-safe using `threading.RLock`.
- Transition attempts perform reflection to identify caller file/line info.

## Schema & API Contracts
- `State` Enum: IDLE, THINKING, PLANNING, RISK_EVALUATION, AWAITING_CONFIRMATION, APPROVED, CANCELLED, ACTING, OBSERVING, REFLECTING, REVIEWING, EXECUTING, COMPLETED, SPEAKING, LISTENING, TRANSCRIBING, ERROR, ABORTED, SHUTDOWN.
- `_ALLOWED_TRANSITIONS` sets defining strict allowed topological paths.
- `StateGuard`: Context manager handling temporary transition bounds with asyncio.CancelledError mapping to ABORTED.

## Dependencies
- Only standard library (`threading`, `enum`, `inspect`, `datetime`, `logging`).

## Configuration Variables
- Trail size capped at `100` elements.

## Prompts
- None.
