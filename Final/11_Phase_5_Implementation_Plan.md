# PHASE 5: CONTROLLED IMPLEMENTATION PLAN

## ACTIVE AGENTS
- Architecture Analysis Agent
- Refactor Planning Agent
- Build System Agent

## EXECUTION PHASES
This must be executed in atomic, isolated steps. Do not batch.

### STEP 1: Interface Definition (Reversible)
1. Create `core/interfaces/controller.py` defining `BaseController` ABC.
2. Run MyPy/Ruff. No runtime changes yet.

### STEP 2: Controller Inheritance (Atomic)
1. Modify `core/controller_v2.py` to inherit from `BaseController`.
2. Implement missing abstract methods (if any).
3. Validate startup behavior.

### STEP 3: Bootstrapping Strictness
1. Modify `core/runtime/entrypoint.py` to cast/type-check `controller` as `BaseController`.
2. Remove `typing.Any`.
3. Validate runtime and shutdown sequences.

### STEP 4: Async Teardown Fixes
1. Modify `entrypoint.py` teardown phase. 
2. Remove `contextlib.suppress(Exception)`.
3. Log all exceptions explicitly.
4. Add `await asyncio.gather(*pending_tasks, return_exceptions=True)` to force cleanup.
5. Validate memory leaks using psutil across rapid restarts.

### STEP 5: Blocking Call Audit
1. Audit `core/state_machine.py`.
2. Replace synchronous I/O.
3. Stress test state transitions.

**Rollback Plan**: Git commit after each atomic step. If health checks fail, revert to previous commit.
