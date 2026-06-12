# Analysis Report for wake_word.py

## Dependencies
- __future__.annotations
- asyncio
- logging
- os
- threading
- typing.Any
- typing.Callable
- typing.Optional

## Schemas
- WakeWordDetector

## API Contracts
- WakeWordDetector.__init__(self, config, loop, on_wake, on_cancel)
- WakeWordDetector._get(self, key, default)
- WakeWordDetector._fire_wake(self)
- WakeWordDetector._fire_cancel(self)
- WakeWordDetector._wait_blocking(self)
- WakeWordDetector.stop(self)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Wake-word detection with porcupine and continuous-listen fallback.

The WakeWordDetector class supports the V2 acceptance test API:
  WakeWordDetector(config, loop, on_wake, on_cancel)
  detector._wake_word     — str
  detector._cancel_words  — set[str]
  detector._fire_wake()   — trigger on_wake callback
  detector._fire_cancel() — trigger on_cancel callback
  detector.stop()         — halt detection thread

