# API Analyst Report: autonomy\scheduler.py

## Dependencies
- `from __future__ import annotations`
- `import threading`
- `import uuid`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from datetime import datetime`
- `from datetime import timedelta`
- `from datetime import timezone`
- `from enum import Enum`
- `from typing import Optional`

## Configuration Variables
- `WAITING` = `'waiting'`
- `DUE` = `'due'`
- `RUNNING` = `'running'`
- `COMPLETED` = `'completed'`
- `CANCELLED` = `'cancelled'`

## Schemas & API Contracts (Classes)

### Class `ScheduleStatus(str, Enum)`


### Class `ScheduledMission`
> Entry in the scheduler queue.

**Fields/Schema:**
  - `entry_id: str`
  - `mission_id: str`
  - `goal_id: str`
  - `run_at: datetime`
  - `status: ScheduleStatus`
  - `attempt_number: int`
  - `max_attempts: int`
  - `base_delay_seconds: float`
  - `backoff_factor: float`
  - `description: str`
  - `created_at: datetime`
  - `last_run_at: Optional[datetime]`
  - `completed_at: Optional[datetime]`

**Methods:**
- @property
- `def is_due(self) -> bool`
- @property
- `def next_retry_delay(self) -> float`
  - *Seconds to wait before the next retry attempt.*
- `def mark_completed(self) -> None`
- `def mark_cancelled(self) -> None`
- `def schedule_retry(self) -> bool`
  - *Advance the attempt counter and set a new run_at.*
- `def to_dict(self) -> dict`


### Class `Scheduler`
> Pull-based mission scheduler with exponential back-off.

Usage:
    scheduler = Scheduler()
    scheduler.enqueue(mission_id="abc", goal_id="xyz", delay_seconds=0)

    # In your main loop:
    for entry in scheduler.due():
        entry.mark_running()
        run_mission(entry.mission_id)
        entry.mark_completed()

**Methods:**
- `def __init__(self) -> None`
- `def enqueue(self, mission_id: str, goal_id: str, delay_seconds: float=0.0, max_attempts: int=3, base_delay_seconds: float=30.0, backoff_factor: float=2.0, description: str='') -> ScheduledMission`
- `def due(self) -> list[ScheduledMission]`
  - *Return all entries that are currently due, sorted by run_at.*
- `def get(self, entry_id: str) -> ScheduledMission`
- `def cancel(self, entry_id: str) -> None`
- `def pending(self) -> list[ScheduledMission]`
- `def snapshot(self) -> list[dict]`
- `def restore(self, data: list[dict]) -> None`


## Functions & Endpoints

### `_utcnow`
`def _utcnow() -> datetime`