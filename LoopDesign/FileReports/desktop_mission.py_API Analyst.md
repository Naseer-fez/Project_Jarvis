# API Analyst Report: desktop\mission.py

## Dependencies
- `from __future__ import annotations`
- `import inspect`
- `import time`
- `import uuid`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from enum import Enum`
- `from typing import Any`
- `from typing import Callable`
- `from typing import Iterable`
- `from core.desktop.actions import DesktopActionExecutor`
- `from core.desktop.contracts import ApprovalDecision`
- `from core.desktop.contracts import DesktopAction`
- `from core.desktop.contracts import DesktopActionResult`
- `from core.desktop.contracts import DesktopActionStatus`
- `from core.desktop.contracts import DesktopChange`
- `from core.desktop.contracts import DesktopObservation`
- `from core.desktop.observation import DesktopObserver`

## Configuration Variables
- `RUNNING` = `'running'`
- `SUCCEEDED` = `'succeeded'`
- `FAILED` = `'failed'`
- `NEEDS_USER` = `'needs_user'`
- `STOPPED` = `'stopped'`
- `NONE` = `'none'`
- `RETRY` = `'retry'`
- `REOBSERVE` = `'reobserve'`
- `ASK_USER` = `'ask_user'`
- `STOP` = `'stop'`

## Schemas & API Contracts (Classes)

### Class `DesktopMissionStatus(str, Enum)`


### Class `RecoveryDecision(str, Enum)`


### Class `MissionStepRecord`
**Fields/Schema:**
  - `step_id: str`
  - `action: dict[str, Any]`
  - `observation_before: dict[str, Any] | None`
  - `approval: dict[str, Any] | None`
  - `result: dict[str, Any] | None`
  - `observation_after: dict[str, Any] | None`
  - `change: dict[str, Any] | None`
  - `recovery_decision: str`
  - `attempts: int`
  - `status: str`
  - `error: str`

**Methods:**
- `def to_dict(self) -> dict[str, Any]`


### Class `MissionExecutionRecord`
**Fields/Schema:**
  - `goal: str`
  - `plan: list[dict[str, Any]]`
  - `mission_id: str`
  - `status: DesktopMissionStatus`
  - `steps: list[MissionStepRecord]`
  - `final_summary: str`
  - `started_at: float`
  - `ended_at: float | None`
  - `metadata: dict[str, Any]`

**Methods:**
- `def close(self, status: DesktopMissionStatus, summary: str) -> None`
- @property
- `def duration_seconds(self) -> float`
- `def explain(self) -> str`
- `def to_dict(self) -> dict[str, Any]`


### Class `DesktopMissionExecutor`
> Observe, act, verify, recover, and explain desktop missions.

**Methods:**
- `def __init__(self, *, action_executor: DesktopActionExecutor | None=None, observer: DesktopObserver | None=None, approval_callback: Callable[[DesktopAction, str], Any] | None=None, audit_writer: Callable[[str, dict[str, Any]], str] | None=None, max_retries: int=1, min_confidence: float=0.35) -> None`
- `async def run(self, *, goal: str, actions: Iterable[DesktopAction], plan_summary: str='') -> MissionExecutionRecord`
- `async def _observe_with_recovery(self, step: MissionStepRecord, label: str) -> DesktopObservation`
- `async def _execute_with_verification(self, step: MissionStepRecord, action: DesktopAction, before: DesktopObservation, approval: ApprovalDecision) -> bool`
- `async def _approval_if_required(self, action: DesktopAction) -> ApprovalDecision`
- `async def _approval(self, action: DesktopAction, reason: str) -> ApprovalDecision`
- `def _summary_for(self, record: MissionExecutionRecord) -> str`
- `def _audit(self, event_type: str, payload: dict[str, Any]) -> None`

