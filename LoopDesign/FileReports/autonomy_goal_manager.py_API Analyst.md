# API Analyst Report: autonomy\goal_manager.py

## Dependencies
- `from __future__ import annotations`
- `import threading`
- `import uuid`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from datetime import datetime`
- `from datetime import timezone`
- `from enum import Enum`
- `from typing import Optional`

## Configuration Variables
- `PENDING` = `'pending'`
- `ACTIVE` = `'active'`
- `PAUSED` = `'paused'`
- `COMPLETED` = `'completed'`
- `FAILED` = `'failed'`
- `CANCELLED` = `'cancelled'`

## Schemas & API Contracts (Classes)

### Class `GoalStatus(str, Enum)`


### Class `Goal`
> A single long-lived agent objective.

**Fields/Schema:**
  - `goal_id: str`
  - `description: str`
  - `priority: int`
  - `status: GoalStatus`
  - `parent_goal_id: Optional[str]`
  - `metadata: dict`
  - `created_at: datetime`
  - `started_at: Optional[datetime]`
  - `completed_at: Optional[datetime]`
  - `deadline: Optional[datetime]`
  - `outcome: Optional[str]`

**Methods:**
- `def start(self) -> None`
- `def complete(self, outcome: str='') -> None`
- `def fail(self, reason: str='') -> None`
- `def cancel(self, reason: str='') -> None`
- `def pause(self) -> None`
- `def resume(self) -> None`
- @property
- `def is_terminal(self) -> bool`
- `def to_dict(self) -> dict`


### Class `GoalManager`
> Registry and lifecycle manager for all agent goals.

Usage:
    gm = GoalManager()
    gid = gm.create_goal("Summarise all emails from today", priority=2)
    gm.start_goal(gid)
    ...
    gm.complete_goal(gid, outcome="12 emails summarised")

**Methods:**
- `def __init__(self) -> None`
- `def create_goal(self, description: str, priority: int=5, parent_goal_id: Optional[str]=None, deadline: Optional[datetime]=None, metadata: Optional[dict]=None) -> str`
- `def get_goal(self, goal_id: str) -> Goal`
- `def start_goal(self, goal_id: str) -> None`
- `def complete_goal(self, goal_id: str, outcome: str='') -> None`
- `def fail_goal(self, goal_id: str, reason: str='') -> None`
- `def cancel_goal(self, goal_id: str, reason: str='') -> None`
- `def pause_goal(self, goal_id: str) -> None`
- `def resume_goal(self, goal_id: str) -> None`
- `def update_goal(self, goal_id: str, description: Optional[str]=None, priority: Optional[int]=None, deadline: Optional[datetime]=None, metadata: Optional[dict]=None) -> None`
- `def remove_goal(self, goal_id: str) -> None`
- `def next_goal(self) -> Optional[Goal]`
  - *Return the highest-priority pending or paused goal.*
- `def active_goals(self) -> list[Goal]`
- `def all_goals(self) -> list[Goal]`
- `def get_goals_by_status(self, status: GoalStatus) -> list[Goal]`
- `def get_subgoals(self, parent_goal_id: str) -> list[Goal]`
- `def snapshot(self) -> list[dict]`
- `def restore(self, data: list[dict]) -> None`
  - *Reload goals from a persisted snapshot (e.g. after restart).*


## Functions & Endpoints

### `_utcnow`
`def _utcnow() -> datetime`