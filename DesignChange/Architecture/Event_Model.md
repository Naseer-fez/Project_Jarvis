# Event Model

The system utilizes an internal pub/sub event bus (`core.runtime.event_bus`) to broadcast state changes across isolated subsystems.

## Event Payloads
Events are typically typed Dataclasses or dictionaries containing:
- `event_type`: The topic.
- `payload`: Contextual data (e.g., token usage stats, goal transitions).
- `timestamp`: UTC issuance time.

## Key Subscriptions
- **`JarvisState` Transitions**: Fired whenever the agent moves from `Idle` -> `Thinking` -> `Executing`. The `dashboard` subscribes to this to render the UI.
- **`GoalStatus` Updates**: Fired when a subtask completes or fails, alerting the `AgentLoopEngine` to reflect and re-plan.
- **LLM Telemetry**: Emitted by `core.llm.telemetry.RoutingTelemetry` whenever a token is consumed. Subscribed by `dashboard` to show live billing dashboards.
- **Hardware Interrupts**: Fired by `core.hardware.serial_controller` or `WakeWordDetector` to violently interrupt the agent loop and force attention to physical inputs.