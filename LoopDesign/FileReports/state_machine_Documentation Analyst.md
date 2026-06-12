# Analysis Report for state_machine.py

## Dependencies
- __future__.annotations
- inspect
- logging
- threading
- datetime.datetime
- enum.Enum
- typing.Callable
- typing.Any

## Schemas
- IllegalTransitionError
- State
- StateGuard
- StateMachine

## API Contracts
- StateGuard.__init__(self, state_machine, target_state)
- StateGuard.__enter__(self)
- StateGuard.__exit__(self, exc_type, exc_val, exc_tb)
- StateMachine.__init__(self, event_bus)
- StateMachine.state(self)
- StateMachine.add_listener(self, listener)
- StateMachine.remove_listener(self, listener)
- StateMachine.can_transition(self, new_state)
- StateMachine.get_valid_transitions(self, state)
- StateMachine.get_transition_graph(self)
- StateMachine._notify(self, old_state, new_state)
- StateMachine.transition(self, new_state)
- StateMachine.reset(self)
- StateMachine.force_idle(self)
- StateMachine.transition_to(self, target_state)
- StateMachine.__enter__(self)
- StateMachine.__exit__(self, exc_type, exc_val, exc_tb)

## Configuration Variables
- _ALLOWED_TRANSITIONS (typed)

## Assumptions & Notes
- Module Docstring: Finite-state machine used across legacy and current Jarvis flows.

