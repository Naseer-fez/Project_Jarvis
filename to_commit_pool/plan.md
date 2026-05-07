# Jarvis Closed-Loop Reliability Plan

## Objective

The next goal is not to add more integrations or flashy standalone features. The next goal is to make Jarvis safely complete real tasks end to end.

Jarvis already has many capable pieces: runtime modes, memory, model routing, tools, desktop control, dashboard, safety checks, audit logging, and integrations. The project now needs one cohesive operator architecture that turns those pieces into a dependable assistant.

The target loop is:

```text
observe -> decide -> act -> verify -> recover -> explain
```

## Biggest Problem

Jarvis's biggest problem is closed-loop reliability.

It can start, route models, store memory, call tools, and perform some desktop actions. But it does not yet consistently prove that an action worked, recover when the UI drifts, or explain what happened in one unified record.

The risk is not that Jarvis has too few features. The risk is that too many capable subsystems exist without one reliable execution loop. More features will make the system harder to trust unless actions, observations, safety decisions, retries, and outcomes are joined into one predictable flow.

## What The Project Lacks

### Unified Desktop Action Contract

Jarvis needs one normalized `DesktopAction` interface for desktop operations:

- launch app
- focus window
- move mouse
- click and double-click
- scroll
- drag
- type text
- press hotkeys
- use clipboard
- capture action metadata
- route all desktop actions through risk and audit checks

Desktop execution should stop feeling like separate one-off helper calls. Higher-level planners should consume one action contract with consistent inputs, results, risk metadata, and failure states.

### Unified Screen Observation Contract

Jarvis needs one normalized `DesktopObservation` interface for understanding what is visible:

- screenshot capture
- active window metadata
- OCR results
- detected UI targets
- confidence scores
- before/after comparison
- change detection
- low-confidence explanations

The system should avoid blind coordinate clicking. It should observe the screen, act only when confidence is acceptable, then observe again to verify whether the expected change happened.

### Planner-Executor-Recovery Mission Loop

Jarvis needs a real mission loop, not just isolated tool calls.

Each mission should track:

- user goal
- generated plan
- selected action
- observation before action
- approval decision
- execution result
- observation after action
- detected error or no-op
- retry or recovery choice
- final result summary

This should become a `MissionExecutionRecord` so Jarvis can explain what it did, why it did it, where it failed, and what it needs from the user.

### Product-Level Safety And Approval UX

Jarvis has safety checks, but it still needs a clearer user-facing safety model:

- visible action feed
- approvals for risky actions
- emergency stop
- audit review
- autonomy mode display
- explicit sensitive-surface policy
- clear pause behavior before destructive, admin, payment, account, or password-related actions

Safety should be a product behavior, not only internal code paths.

### Reduced Legacy Overlap

The repo still contains legacy modules and overlapping paths from earlier phases. Consolidation is needed, but it should happen only after behavior is covered by tests.

The priority is not cosmetic cleanup. The priority is reducing parallel execution paths that make it unclear which subsystem is authoritative.

## Implementation Order

### 1. Finish Reliable Runtime Foundation

Keep Phase 1 production readiness as the foundation:

- one supported startup path through `main.py`
- predictable config loading
- project-root-relative runtime paths
- explicit health checks
- production guardrails
- clean wrapper scripts
- stable logging and audit output
- focused runtime and smoke tests

Do not build deeper autonomy on top of an unstable runtime.

### 2. Build Normalized Desktop Action Layer

Introduce the `DesktopAction` contract and route all desktop operations through it.

Required outcome:

- one interface for launch, focus, click, type, hotkey, scroll, drag, clipboard, and screenshot-adjacent action metadata
- consistent action result shape
- risk tier attached to every action
- audit event emitted for every action
- failures represented as structured results, not only exceptions or logs

### 3. Build Normalized Screen Observation Layer

Introduce the `DesktopObservation` contract and make visual grounding reusable.

Required outcome:

- one interface for screenshot, OCR, active-window context, visible targets, confidence, and change detection
- observation can run before and after an action
- observation results can be attached to mission records
- low confidence causes retry, re-observation, confirmation request, or safe stop

### 4. Add Mission Execution Loop

Build a planner-executor-recovery loop around actions and observations.

Required outcome:

- Jarvis can turn a bounded user goal into steps
- each step has an observation, action, verification, and recovery decision
- failed or no-op actions are detected
- retry limits prevent loops
- user help is requested when confidence stays low
- final response explains what was completed and what was not

### 5. Add Visible Safety Controls And Audit Review

Make safety understandable to the user.

Required outcome:

- current autonomy mode is visible
- risky actions ask for approval
- stop/cancel behavior is consistent
- sensitive surfaces are handled by stricter policy
- action history is reviewable
- audit logs include goal, plan, action, observation, approval, error, retry, and final result

### 6. Consolidate Legacy And Parallel Paths

Only after behavior is covered by tests, reduce overlapping modules and old execution paths.

Required outcome:

- remove or deprecate duplicate helper paths
- keep compatibility shims only where tests or public usage require them
- update docs to describe the authoritative path
- avoid broad rewrites that do not improve closed-loop reliability

## Acceptance Gates

- Jarvis can perform one desktop action through a single action interface.
- Jarvis can observe screen state before and after an action.
- Jarvis can detect failed or no-op actions.
- Jarvis can retry, re-observe, ask for help, or stop when confidence is low.
- Jarvis records goal, plan, action, observation, approval, error, retry, and result.
- Jarvis explains what happened after a mission finishes.
- Risky actions still require confirmation.
- Sensitive actions remain blocked or paused behind stricter policy.
- Existing runtime tests remain green.
- Existing smoke tests remain green.
- Existing model-routing tests remain green.

## Test Plan

This plan file is documentation-only, so no code tests are required after saving it.

For implementation work that follows this plan, use this validation order:

1. `.\run-tests.ps1 tests/test_runtime_phase1.py -q`
2. `.\run-tests.ps1 tests/test_smoke.py -q`
3. `.\run-tests.ps1 tests/test_runtime_model_routing.py -q`
4. Focused tests for new desktop action, observation, and mission-loop behavior
5. Full suite with `.\run-tests.ps1 -q` before declaring a phase complete

## Assumptions

- `plan.md` is the immediate engineering guide for the next implementation phase.
- `JARVIS_FUTURE.md` remains the long-term roadmap.
- The next implementation priority is closed-loop reliability, not more standalone integrations.
- Phase 1 production readiness remains necessary, but Phase 2 and Phase 3 closed-loop behavior should drive the next work.
- Jarvis is Windows-first.
- Direct text control, voice control, hotkey control, dashboard control, and future browser control should all converge on the same observe-decide-act-verify-recover-explain loop.
