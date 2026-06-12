# Analysis Report for event_bus.py

## Dependencies
- __future__.annotations
- asyncio
- logging
- threading
- time
- uuid
- collections.deque
- dataclasses.dataclass
- dataclasses.field
- typing.Any
- typing.Awaitable
- typing.Callable
- typing.Union

## Schemas
- EventRecord
- EventRecord attribute: event_id
- EventRecord attribute: event_type
- EventRecord attribute: payload
- EventRecord attribute: source
- EventRecord attribute: created_at
- EventBus

## API Contracts
- EventRecord.to_dict(self)
- EventBus.__init__(self)
- EventBus._try_capture_loop(self)
- EventBus.subscribe(self, event_type, callback)
- EventBus.unsubscribe(self, event_type, callback)
- EventBus.publish(self, event_type, data)
- EventBus.replay(self, event_type)
- EventBus.clear_history(self)
- EventBus._record(self, event_type, payload)
- EventBus._callbacks_for(self, event_key)
- EventBus._dispatch_callback(self, callback, data, event_key)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Lightweight pub/sub event bus for decoupled component communications.

