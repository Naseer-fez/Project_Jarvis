# Analysis Report for goal_manager.py

## Dependencies
- __future__.annotations
- threading
- uuid
- dataclasses.dataclass
- dataclasses.field
- datetime.datetime
- datetime.timezone
- enum.Enum
- typing.Optional

## Schemas
- GoalStatus
- Goal
- Goal attribute: goal_id
- Goal attribute: description
- Goal attribute: priority
- Goal attribute: status
- Goal attribute: parent_goal_id
- Goal attribute: metadata
- Goal attribute: created_at
- Goal attribute: started_at
- Goal attribute: completed_at
- Goal attribute: deadline
- Goal attribute: outcome
- GoalManager

## API Contracts
- _utcnow()
- Goal.start(self)
- Goal.complete(self, outcome)
- Goal.fail(self, reason)
- Goal.cancel(self, reason)
- Goal.pause(self)
- Goal.resume(self)
- Goal.is_terminal(self)
- Goal.to_dict(self)
- GoalManager.__init__(self)
- GoalManager.create_goal(self, description, priority, parent_goal_id, deadline, metadata)
- GoalManager.get_goal(self, goal_id)
- GoalManager.start_goal(self, goal_id)
- GoalManager.complete_goal(self, goal_id, outcome)
- GoalManager.fail_goal(self, goal_id, reason)
- GoalManager.cancel_goal(self, goal_id, reason)
- GoalManager.pause_goal(self, goal_id)
- GoalManager.resume_goal(self, goal_id)
- GoalManager.update_goal(self, goal_id, description, priority, deadline, metadata)
- GoalManager.remove_goal(self, goal_id)
- GoalManager.next_goal(self)
- GoalManager.active_goals(self)
- GoalManager.all_goals(self)
- GoalManager.get_goals_by_status(self, status)
- GoalManager.get_subgoals(self, parent_goal_id)
- GoalManager.snapshot(self)
- GoalManager.restore(self, data)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: core/autonomy/goal_manager.py

Owns the lifecycle of long-lived agent goals.
A Goal is a high-level desired outcome that may span multiple Missions.

Responsibilities:
- Create / update / complete / cancel goals
- Prioritise active goals
- Query which goal the agent should work on next
- Persist goal state (via snapshot / restore)

