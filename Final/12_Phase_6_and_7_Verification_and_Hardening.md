# PHASE 6 & 7: SYSTEM VERIFICATION & HARDENING

## PHASE 6: DEEP SYSTEM VERIFICATION
**Active Agents**: Testing & Coverage Agent, Regression Detection Agent

### 1. Chaos Runtime Tests
- Inject `asyncio.sleep(5)` arbitrarily in database read calls to verify timeout handling and event loop resilience.
- Randomly kill the dashboard HTTP server and verify if the core controller survives.

### 2. Startup/Shutdown Cycle Stress Test
- Write a script to launch `main.py --headless` and send `SIGINT` repeatedly (100 times) at different intervals (1s, 2s, 5s).
- Monitor RAM and file descriptors (using `psutil` or `lsof`) to ensure zero leakage.

### 3. Concurrency Workload Simulation
- Fire 50 simultaneous plugin events into the state machine.
- Verify that `core.controller_v2` processes them without dropping state or deadlocking.

## PHASE 7: FINAL HARDENING
**Active Agents**: Maintainability Agent, Logging & Observability Agent

### 1. Dead Code Elimination
- Run `vulture` across the entire codebase. Remove unused functions and orphan classes in `plugins/` and `workflows/`.
- Ensure no "placeholder" logic exists.

### 2. Standardization
- Reformat all files with `ruff` and `black`.
- Enforce strict MyPy (`--strict` flag) on `core/`.

### 3. Observability Upgrade
- Inject OpenTelemetry tracing into the newly created `BaseController` and `state_machine.py`.
- Ensure all unstructured print statements (`print()`) are replaced with structured `log.info()` or `log.debug()`.
