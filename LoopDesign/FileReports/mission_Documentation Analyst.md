# Analysis Report for mission.py

## Dependencies
- __future__.annotations
- inspect
- time
- uuid
- dataclasses.dataclass
- dataclasses.field
- enum.Enum
- typing.Any
- typing.Callable
- typing.Iterable
- core.desktop.actions.DesktopActionExecutor
- core.desktop.contracts.ApprovalDecision
- core.desktop.contracts.DesktopAction
- core.desktop.contracts.DesktopActionResult
- core.desktop.contracts.DesktopActionStatus
- core.desktop.contracts.DesktopChange
- core.desktop.contracts.DesktopObservation
- core.desktop.observation.DesktopObserver

## Schemas
- DesktopMissionStatus
- RecoveryDecision
- MissionStepRecord
- MissionStepRecord attribute: step_id
- MissionStepRecord attribute: action
- MissionStepRecord attribute: observation_before
- MissionStepRecord attribute: approval
- MissionStepRecord attribute: result
- MissionStepRecord attribute: observation_after
- MissionStepRecord attribute: change
- MissionStepRecord attribute: recovery_decision
- MissionStepRecord attribute: attempts
- MissionStepRecord attribute: status
- MissionStepRecord attribute: error
- MissionExecutionRecord
- MissionExecutionRecord attribute: goal
- MissionExecutionRecord attribute: plan
- MissionExecutionRecord attribute: mission_id
- MissionExecutionRecord attribute: status
- MissionExecutionRecord attribute: steps
- MissionExecutionRecord attribute: final_summary
- MissionExecutionRecord attribute: started_at
- MissionExecutionRecord attribute: ended_at
- MissionExecutionRecord attribute: metadata
- DesktopMissionExecutor

## API Contracts
- MissionStepRecord.to_dict(self)
- MissionExecutionRecord.close(self, status, summary)
- MissionExecutionRecord.duration_seconds(self)
- MissionExecutionRecord.explain(self)
- MissionExecutionRecord.to_dict(self)
- DesktopMissionExecutor.__init__(self)
- DesktopMissionExecutor._summary_for(self, record)
- DesktopMissionExecutor._audit(self, event_type, payload)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Planner-executor-recovery loop for bounded desktop missions.

