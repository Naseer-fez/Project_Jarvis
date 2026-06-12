# Instability Hotspot Map

## 1. Dynamic Class Loading (`core.runtime.bootstrap`)
- Bootstrapping dynamically resolves `_load_controller_class()`.
- **Symptom**: Startup crashes if modules move or config mismatch occurs. Hard to statically verify.

## 2. Asynchronous Shutdown Sequence (`core.runtime.entrypoint`)
- The teardown phase catches `asyncio.CancelledError` and swallows it during shutdown.
- **Symptom**: Zombie processes, orphaned dashboard listeners, leaked file handles during rapid restarts.

## 3. Global Exception Handlers
- Top-level `Exception` catches in `run_entrypoint` (`main.py`) map to `GENERIC_ERROR`.
- `_install_loop_exception_handler` overrides asyncio defaults, potentially masking deep coroutine crashes.

## 4. State Machine Couplings
- `core/state_machine.py` directly interacts with the controller loop. Unbounded async tasks inside the state machine can starve the main runtime loop.
