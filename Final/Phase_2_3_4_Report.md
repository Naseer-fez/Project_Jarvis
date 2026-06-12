# PHASE 2: ROOT CAUSE ANALYSIS

## 2.1 Diagnostics & Hotspots
Based on the execution of the Runtime Failure and Concurrency Agents against the internal `ISSUE_REGISTRY.md` and static typing traces, the core issues driving instability have been diagnosed.

### Hotspot 1: Zombie Tasks & Inconsistent Teardown (Severity: HIGH)
- **Condition**: `asyncio.CancelledError` is suppressed broadly in `core.runtime.entrypoint`. When the application receives a shutdown signal, pending asynchronous tasks are cancelled, but resource locks are not explicitly released before the exception is swallowed.
- **Root Cause**: Poor async lifecycle management and suppression of `CancelledError`.
- **Regression Risk**: High (Resource Leaks).

### Hotspot 2: Silent Dynamic Resolution Failures (Severity: HIGH)
- **Condition**: `_load_controller_class()` uses string-based import resolution mapped from `jarvis.ini`. If the class is missing or renamed, it throws a generic ImportError at runtime.
- **Root Cause**: Lack of static typing and hardcoded dynamic string loading.
- **Regression Risk**: High (Startup Collapse).

### Hotspot 3: Unbounded Controller Execution Loop (Severity: MEDIUM-HIGH)
- **Condition**: `controller_v2.py` and `state_machine.py` share the main asyncio event loop without strict timeout bounds or execution budget constraints.
- **Root Cause**: Synchronous IO inside a plugin or heavy LLM processing without yield.
- **Regression Risk**: Medium (Thread Starvation).

---

# PHASE 3: RECOVERY STRATEGY GENERATION

## 3.1 Blueprinting Fixes

### Strategy A: Async Lifecycle Management
1. **Explicit Teardown**: Intercept `CancelledError` and ensure graceful release of resources, file descriptors, and sockets.
2. **Execution Engine**: Establish a clear shutdown hook in `core.runtime.entrypoint`.

### Strategy B: Type Safety & Hardening
1. **Type Contracts**: Patch `controller` instance in `entrypoint.py` from `Any` to `BaseController`.
2. **Resolution Failures**: Ensure `_load_controller_class()` implements a resilient fallback mechanism.

---

# PHASE 4: MULTI-AGENT VALIDATION

## 4.1 Validator Agent & Risk Assessment
**Status**: APPROVED by Validator Agent. The revised strategy balances complete structural recovery with safe backward-compatibility layers.
