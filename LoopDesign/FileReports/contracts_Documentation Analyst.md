# Analysis Report for contracts.py

## Dependencies
- __future__.annotations
- time
- uuid
- dataclasses.dataclass
- dataclasses.field
- enum.Enum
- typing.Any

## Schemas
- DesktopActionType
- DesktopRiskTier
- DesktopActionStatus
- DesktopAction
- DesktopAction attribute: action_type
- DesktopAction attribute: params
- DesktopAction attribute: description
- DesktopAction attribute: expected_change
- DesktopAction attribute: risk_tier
- DesktopAction attribute: requires_approval
- DesktopAction attribute: action_id
- DesktopAction attribute: metadata
- DesktopActionResult
- DesktopActionResult attribute: action_id
- DesktopActionResult attribute: action_type
- DesktopActionResult attribute: success
- DesktopActionResult attribute: status
- DesktopActionResult attribute: output
- DesktopActionResult attribute: error
- DesktopActionResult attribute: risk_tier
- DesktopActionResult attribute: audit_hash
- DesktopActionResult attribute: started_at
- DesktopActionResult attribute: ended_at
- DesktopActionResult attribute: metadata
- ScreenTarget
- ScreenTarget attribute: label
- ScreenTarget attribute: x
- ScreenTarget attribute: y
- ScreenTarget attribute: width
- ScreenTarget attribute: height
- ScreenTarget attribute: confidence
- ScreenTarget attribute: metadata
- DesktopObservation
- DesktopObservation attribute: observation_id
- DesktopObservation attribute: screenshot_path
- DesktopObservation attribute: screenshot_fingerprint
- DesktopObservation attribute: active_window
- DesktopObservation attribute: ocr_text
- DesktopObservation attribute: targets
- DesktopObservation attribute: confidence
- DesktopObservation attribute: low_confidence_reason
- DesktopObservation attribute: metadata
- DesktopObservation attribute: captured_at
- DesktopChange
- DesktopChange attribute: changed
- DesktopChange attribute: confidence
- DesktopChange attribute: summary
- DesktopChange attribute: before_observation_id
- DesktopChange attribute: after_observation_id
- DesktopChange attribute: metadata
- ApprovalDecision
- ApprovalDecision attribute: required
- ApprovalDecision attribute: approved
- ApprovalDecision attribute: reason
- ApprovalDecision attribute: mode

## API Contracts
- _new_id(prefix)
- DesktopAction.action_name(self)
- DesktopAction.to_dict(self)
- DesktopActionResult.duration_seconds(self)
- DesktopActionResult.to_dict(self)
- ScreenTarget.to_dict(self)
- DesktopObservation.to_dict(self)
- DesktopChange.to_dict(self)
- ApprovalDecision.to_dict(self)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Contracts for desktop actions, observations, and verification results.

