# Jarvis V3 → Agentic Layer Update

**Version:** Agentic Layer v1.0  
**Date:** 2026-02-23  
**Scope:** Additive only — no existing systems modified

---

## Overview

This update transforms Jarvis from a task executor into a **goal-owning autonomous agent** by adding a new `core/agentic/` module on top of the existing V3 architecture.

No existing files were modified. No planners, dispatchers, or memory internals were touched.

---

## New Module: `core/agentic/`

```
core/agentic/
├── __init__.py          — Module exports
├── goal_manager.py      — Persistent goal ownership & resumption
├── mission.py           — Multi-step mission execution state
├── reflection.py        — Post-execution evaluation & lessons
├── belief_state.py      — Agent confidence & environment beliefs
├── autonomy_policy.py   — Safety rules & confirmation thresholds
├── decision_trace.py    — Human-readable decision audit log
└── scheduler.py         — Deferred goals & retry scheduling
```

---

## File-by-File Changes

### `goal_manager.py`
- Introduces `Goal` and `GoalManager` classes
- Goals persist to `data/agentic/goals.json` across restarts
- States: `pending → active → stalled → completed | aborted`
- `resumable_goals()` re-injects PENDING/STALLED goals after restart
- Atomic save (write-then-rename to prevent corruption)

### `mission.py`
- Introduces `Mission` and `Checkpoint` classes
- Each mission maps to one Goal and tracks discrete steps
- Supports `start()`, `pause()`, `resume()`, `abort()`, `complete()`
- Each mission persists to `data/agentic/missions/<id>.json`
- `human_summary()` gives a readable step-by-step view

### `reflection.py`
- `ReflectionEngine` runs after every mission terminal state
- Evaluates outcome (`success` / `partial` / `failure`)
- Detects known failure patterns (network, timeout, rate limit, auth, not-found)
- Computes belief deltas and calls `BeliefState.update()`
- Writes `ReflectionReport` to hybrid_memory under key `reflection:<mission_id>`

### `belief_state.py`
- `BeliefState` stores confidence scores (0.0–1.0) for:
  - `system_reliability`, `network_reliability`, `agent_confidence`
  - `api_rate_limit_risk`, `risk_tolerance`, `user_interruption_likelihood`
- Beliefs change over time via `update(key, delta)` (clamped to [0,1])
- Persists to `data/agentic/belief_state.json`
- Exposes `should_ask_user()` and `is_reliable_enough()` helpers for policy use

### `autonomy_policy.py`
- `AutonomyPolicy` enforces three-verdict decisions: `ALLOW`, `REQUIRE_CONFIRM`, `DENY`
- **Hard-deny list:** actions like `disable_logging`, `bypass_auth`, `self_modify_policy` — unconditionally forbidden
- **Always-confirm list:** destructive/external actions like `send_email`, `delete_file`, `deploy_to_production`
- **Risk threshold:** risk ≥ 0.6 → confirm; risk ≥ 0.95 → deny
- **Belief-based check:** low agent confidence or low risk tolerance → confirm
- `check_retry()` enforces `max_retries` (default: 3) per action key
- `should_escalate(stall_count)` triggers user escalation after repeated stalls

### `decision_trace.py`
- `DecisionTrace` records every significant agent decision
- Each entry captures: decision text, rationale, confidence %, risk %, rejected alternatives
- Persists as append-only JSON Lines to `data/agentic/traces/<goal_id>.jsonl`
- `print_full_trace()` returns a complete human-readable audit log
- `load_for_goal()` reconstructs history from disk for any past goal

### `scheduler.py`
- `AgentScheduler` manages time-based triggers without background threads
- Trigger types: `once`, `retry`, `recurring`, `check_back`
- `tick()` is called by the agent loop — returns list of due goal_ids
- `schedule_retry()` respects `max_retries` and won't add more after exhaustion
- Persists to `data/agentic/schedule.json` atomically

---

## Architecture: How It Connects

```
User / Voice
    ↓
GoalManager.add_goal()
    ↓
Mission (checkpoints created from planner output)
    ↓
Existing Planner  ←── no changes
    ↓
Existing Dispatcher  ←── no changes
    ↓
Tools / Integrations  ←── no changes
    ↓
ReflectionEngine.reflect(mission)
    ↓
BeliefState.update()  +  HybridMemory.store()
    ↺ feeds back into GoalManager (stall / complete / retry via Scheduler)
```

**AutonomyPolicy** is consulted before any high-risk step is dispatched.  
**DecisionTrace** is written whenever the agent makes a non-trivial choice.

---

## Data Directory Layout

```
data/agentic/
├── goals.json              — All goals (GoalManager)
├── belief_state.json       — Current beliefs (BeliefState)
├── schedule.json           — Pending scheduled triggers (AgentScheduler)
├── missions/
│   └── <mission_id>.json   — One file per mission (Mission)
└── traces/
    └── <goal_id>.jsonl     — Append-only decision log (DecisionTrace)
```

---

## Acceptance Criteria Status

| Requirement | Implemented by |
|---|---|
| Hold a goal across sessions | `GoalManager` + JSON persistence |
| Resume after restart | `GoalManager.resumable_goals()` |
| Explain decisions | `DecisionTrace.print_full_trace()` |
| Reflect and adapt | `ReflectionEngine` + `BeliefState` |
| Enforce safety rules | `AutonomyPolicy` (deny/confirm lists) |
| Ask for confirmation when needed | `PolicyVerdict.REQUIRE_CONFIRM` |
| Abort safely | `Mission.abort()` + `GoalManager.transition(ABORTED)` |

---

## Integration Notes

### Hooking into the existing planner
After calling the planner, wrap its output in a `Mission`:
```python
mission = Mission(goal_id=goal.goal_id, title=goal.description)
for step in planner_output.steps:
    mission.add_checkpoint(step.name, step.description)
mission.save()
goal_manager.link_mission(goal.goal_id, mission.mission_id)
```

### Checking autonomy before dispatch
```python
policy = AutonomyPolicy(belief_state)
decision = policy.evaluate(action_name, risk_score=risk_evaluator.score(action))
if decision.is_denied():
    mission.abort(decision.reason)
elif decision.requires_confirmation():
    # pause and ask user
    mission.pause(decision.reason)
elif decision.is_allowed():
    dispatcher.dispatch(action)
```

### Running reflection after a mission ends
```python
engine = ReflectionEngine(belief_state, memory=hybrid_memory)
report = engine.reflect(mission)
# report.outcome is "success" | "partial" | "failure"
```

### Agent restart sequence
```python
goal_manager = GoalManager(); goal_manager.load()
scheduler = AgentScheduler(); scheduler.load()
belief_state = BeliefState(); belief_state.load()

for goal in goal_manager.resumable_goals():
    # Re-inject into planner
    planner.queue(goal)
```

### Scheduler tick (add to agent main loop)
```python
due_goal_ids = scheduler.tick()
for gid in due_goal_ids:
    goal_manager.transition(gid, GoalStatus.PENDING)
```

---

## What Was NOT Changed

- `core/planner/` — untouched
- `core/dispatcher/` — untouched  
- `core/memory/hybrid_memory.py` — untouched (only called via its public API)
- `core/voice/` — untouched
- `integrations/` — untouched
- All existing tests — unaffected (no monkey-patching, no side effects)

---

## Dependencies

No new third-party packages required. All files use Python standard library only (`json`, `pathlib`, `logging`, `dataclasses`, `enum`, `uuid`, `datetime`).
