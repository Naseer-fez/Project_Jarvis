# PHASE 2: ROOT CAUSE ANALYSIS

## ACTIVE AGENTS
- Runtime Failure Analysis Agent
- Static Analysis Agent
- Async/Concurrency Analysis Agent

## ISSUE 1: Zombie Tasks & Inconsistent Teardown
- **Root Cause**: `asyncio.CancelledError` is suppressed broadly in `core.runtime.entrypoint`. When the application receives a shutdown signal, pending asynchronous tasks are cancelled, but resource locks (e.g., file descriptors, sockets) held by those tasks are not explicitly released before the exception is swallowed.
- **Affected Systems**: `core.runtime`, background plugin threads, dashboard HTTP server.
- **Severity**: High (Resource Leaks).
- **Failure Conditions**: Rapid restarts, ungraceful exits, out-of-memory under load.

## ISSUE 2: Silent Dynamic Resolution Failures
- **Root Cause**: `_load_controller_class()` uses string-based import resolution mapped from `jarvis.ini`. If the class is missing or renamed, it throws a generic ImportError at runtime. MyPy and Ruff cannot validate string-based imports statically.
- **Affected Systems**: `core.runtime.bootstrap`, all dependent modules.
- **Severity**: High (Startup Collapse).
- **Failure Conditions**: Configuration drift, incomplete code refactoring.

## ISSUE 3: Unbounded Controller Execution Loop
- **Root Cause**: `controller_v2.py` and `state_machine.py` share the main asyncio event loop without strict timeout bounds or execution budget constraints. Long-running AI inferences or synchronous blockages in plugins will stall the entire event loop.
- **Affected Systems**: `core.controller`, `core.state_machine`, `plugins`.
- **Severity**: Medium-High (Thread Starvation).
- **Failure Conditions**: Synchronous IO inside a plugin, heavy LLM processing without yield.

## ISSUE 4: Missing Type Contracts on Core Interfaces
- **Root Cause**: `controller` instance in `entrypoint.py` is typed as `Any`. Methods like `run_cli` and `shutdown` are duck-typed.
- **Affected Systems**: `core.runtime`, `core.controller_v2`.
- **Severity**: Medium (Maintainability).
- **Failure Conditions**: Future developers changing the Controller API without updating the entrypoint.
