# Analysis Report for scheduler.py

## Dependencies
- __future__.annotations
- threading
- uuid
- dataclasses.dataclass
- dataclasses.field
- datetime.datetime
- datetime.timedelta
- datetime.timezone
- enum.Enum
- typing.Optional

## Schemas
- ScheduleStatus
- ScheduledMission
- ScheduledMission attribute: entry_id
- ScheduledMission attribute: mission_id
- ScheduledMission attribute: goal_id
- ScheduledMission attribute: run_at
- ScheduledMission attribute: status
- ScheduledMission attribute: attempt_number
- ScheduledMission attribute: max_attempts
- ScheduledMission attribute: base_delay_seconds
- ScheduledMission attribute: backoff_factor
- ScheduledMission attribute: description
- ScheduledMission attribute: created_at
- ScheduledMission attribute: last_run_at
- ScheduledMission attribute: completed_at
- Scheduler

## API Contracts
- _utcnow()
- ScheduledMission.is_due(self)
- ScheduledMission.next_retry_delay(self)
- ScheduledMission.mark_completed(self)
- ScheduledMission.mark_cancelled(self)
- ScheduledMission.schedule_retry(self)
- ScheduledMission.to_dict(self)
- Scheduler.__init__(self)
- Scheduler.enqueue(self, mission_id, goal_id, delay_seconds, max_attempts, base_delay_seconds, backoff_factor, description)
- Scheduler.due(self)
- Scheduler.get(self, entry_id)
- Scheduler.cancel(self, entry_id)
- Scheduler.pending(self)
- Scheduler.snapshot(self)
- Scheduler.restore(self, data)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: core/autonomy/scheduler.py

Manages delayed and retry execution of Missions.

Responsibilities:
- Queue missions for execution at a future time
- Implement exponential back-off for retries
- Expose the next due mission (pull model — no background threads)
- Persist schedule across restarts

Design note:
  This is a *pull-based* scheduler.  The caller (main loop / dispatcher)
  asks `scheduler.due()` on each tick.  There are no background threads,
  no asyncio tasks, no hidden loops — exactly as the spec requires.

