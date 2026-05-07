from __future__ import annotations

import pytest

from core.desktop import (
    DesktopAction,
    DesktopActionExecutor,
    DesktopActionStatus,
    DesktopActionType,
    DesktopMissionExecutor,
    DesktopMissionStatus,
    DesktopObservation,
    DesktopObserver,
    DesktopRiskTier,
)


def _audit_sink(events):
    def _write(event_type, payload):
        events.append((event_type, payload))
        return f"hash-{len(events)}"

    return _write


def _observation(title: str, fingerprint: str, *, confidence: float = 0.9) -> DesktopObservation:
    return DesktopObservation(
        screenshot_path=f"outputs/screenshots/{fingerprint}.png",
        screenshot_fingerprint=fingerprint,
        active_window={"title": title},
        confidence=confidence,
    )


class SequenceObserver(DesktopObserver):
    def __init__(self, observations):
        super().__init__()
        self._observations = list(observations)

    async def observe(self, label: str = "") -> DesktopObservation:
        assert self._observations, f"unexpected observe({label}) call"
        obs = self._observations.pop(0)
        return DesktopObservation(
            screenshot_path=obs.screenshot_path,
            screenshot_fingerprint=obs.screenshot_fingerprint,
            active_window=obs.active_window,
            ocr_text=obs.ocr_text,
            targets=obs.targets,
            confidence=obs.confidence,
            low_confidence_reason=obs.low_confidence_reason,
            metadata={**obs.metadata, "label": label},
        )


@pytest.mark.asyncio
async def test_desktop_action_executor_attaches_risk_and_audit():
    events = []
    calls = []

    async def launch_handler(target, args=None):
        calls.append((target, args))
        return {"success": True, "data": {"target": target, "args": args}}

    executor = DesktopActionExecutor(
        audit_writer=_audit_sink(events),
        action_handlers={DesktopActionType.LAUNCH_APP: launch_handler},
    )
    action = DesktopAction(
        action_type=DesktopActionType.LAUNCH_APP,
        params={"target": "notepad.exe", "args": None},
        requires_approval=False,
    )

    result = await executor.execute(action)

    assert result.success is True
    assert result.status == DesktopActionStatus.SUCCESS
    assert result.risk_tier == DesktopRiskTier.CONFIRM
    assert result.audit_hash == "hash-1"
    assert calls == [("notepad.exe", None)]
    assert events[0][0] == "desktop_action"
    assert events[0][1]["action"]["action_type"] == "launch_application"


@pytest.mark.asyncio
async def test_sensitive_type_text_is_blocked_before_handler_runs():
    events = []
    calls = []

    async def type_handler(text):
        calls.append(text)
        return {"success": True}

    executor = DesktopActionExecutor(
        audit_writer=_audit_sink(events),
        action_handlers={DesktopActionType.TYPE_TEXT: type_handler},
    )
    action = DesktopAction(
        action_type=DesktopActionType.TYPE_TEXT,
        params={"text": "my password is hunter2"},
        requires_approval=False,
    )

    result = await executor.execute(action)

    assert result.success is False
    assert result.status == DesktopActionStatus.BLOCKED
    assert "Sensitive text" in result.error
    assert calls == []
    assert events[0][0] == "desktop_action"


def test_observer_detects_no_op_and_changed_screen():
    observer = DesktopObserver()
    before = _observation("Desktop", "same")
    same_after = _observation("Desktop", "same")
    changed_after = _observation("Notepad", "different")

    no_change = observer.compare(before, same_after)
    changed = observer.compare(before, changed_after)

    assert no_change.changed is False
    assert "No observable" in no_change.summary
    assert changed.changed is True
    assert "screenshot changed" in changed.summary
    assert "active window changed" in changed.summary


@pytest.mark.asyncio
async def test_mission_records_observe_act_verify_cycle():
    events = []
    calls = []

    async def launch_handler(target, args=None):
        calls.append((target, args))
        return {"success": True, "data": {"target": target}}

    action = DesktopAction(
        action_type=DesktopActionType.LAUNCH_APP,
        params={"target": "notepad.exe", "args": None},
        expected_change="active window changes",
        requires_approval=False,
    )
    mission = DesktopMissionExecutor(
        action_executor=DesktopActionExecutor(
            audit_writer=_audit_sink(events),
            action_handlers={DesktopActionType.LAUNCH_APP: launch_handler},
        ),
        observer=SequenceObserver(
            [
                _observation("Desktop", "before"),
                _observation("Notepad", "after"),
            ]
        ),
        audit_writer=_audit_sink(events),
        max_retries=0,
    )

    record = await mission.run(
        goal="open notepad",
        actions=[action],
        plan_summary="Open Notepad.",
    )

    assert record.status == DesktopMissionStatus.SUCCEEDED
    assert "1/1 desktop step(s) verified" in record.final_summary
    assert calls == [("notepad.exe", None)]
    step = record.steps[0]
    assert step.observation_before["active_window"]["title"] == "Desktop"
    assert step.result["success"] is True
    assert step.observation_after["active_window"]["title"] == "Notepad"
    assert step.change["changed"] is True
    assert {event[0] for event in events} >= {
        "desktop_action",
        "desktop_mission_started",
        "desktop_mission_step",
        "desktop_mission_finished",
    }


@pytest.mark.asyncio
async def test_mission_retries_when_expected_change_is_not_observed():
    calls = []

    async def click_handler(x, y):
        calls.append((x, y))
        return {"success": True, "data": {"x": x, "y": y}}

    action = DesktopAction(
        action_type=DesktopActionType.CLICK,
        params={"x": 10, "y": 20},
        expected_change="button reacts",
        requires_approval=False,
    )
    mission = DesktopMissionExecutor(
        action_executor=DesktopActionExecutor(
            audit_writer=_audit_sink([]),
            action_handlers={DesktopActionType.CLICK: click_handler},
        ),
        observer=SequenceObserver(
            [
                _observation("Dialog", "same"),
                _observation("Dialog", "same"),
                _observation("Dialog", "changed"),
            ]
        ),
        audit_writer=_audit_sink([]),
        max_retries=1,
    )

    record = await mission.run(goal="click continue", actions=[action])

    assert record.status == DesktopMissionStatus.SUCCEEDED
    assert calls == [(10, 20), (10, 20)]
    step = record.steps[0]
    assert step.attempts == 2
    assert step.recovery_decision == "retry"
    assert step.change["changed"] is True


@pytest.mark.asyncio
async def test_mission_pauses_for_unapproved_risky_action():
    calls = []

    async def click_handler(x, y):
        calls.append((x, y))
        return {"success": True}

    action = DesktopAction(
        action_type=DesktopActionType.CLICK,
        params={"x": 10, "y": 20},
    )
    mission = DesktopMissionExecutor(
        action_executor=DesktopActionExecutor(
            audit_writer=_audit_sink([]),
            action_handlers={DesktopActionType.CLICK: click_handler},
        ),
        observer=SequenceObserver([_observation("Dialog", "before")]),
        audit_writer=_audit_sink([]),
    )

    record = await mission.run(goal="click continue", actions=[action])

    assert record.status == DesktopMissionStatus.NEEDS_USER
    assert calls == []
    assert record.steps[0].approval["required"] is True
    assert record.steps[0].recovery_decision == "ask_user"
