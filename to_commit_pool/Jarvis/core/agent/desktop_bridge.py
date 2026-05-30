"""Bridge between the agent loop and the desktop mission executor.

Converts agent-loop step dicts ({action, params}) into DesktopAction
objects, runs them through the observe-act-verify-recover cycle via
DesktopMissionExecutor, and returns results the agent loop can consume.
"""

from __future__ import annotations

import logging
from typing import Any

from core.desktop.actions import DesktopActionExecutor
from core.desktop.contracts import (
    DesktopAction,
    DesktopActionType,
)
from core.desktop.mission import (
    DesktopMissionExecutor,
    DesktopMissionStatus,
    MissionExecutionRecord,
)
from core.desktop.observation import DesktopObserver
from core.tools.tool_router import ToolObservation

logger = logging.getLogger("Jarvis.DesktopBridge")


# Actions that should be routed through the desktop observe→act→verify loop.
DESKTOP_TOOL_NAMES: frozenset[str] = frozenset({
    "click",
    "double_click",
    "right_click",
    "click_text_on_screen",
    "click_screen_target",
    "double_click_screen_target",
    "right_click_screen_target",
    "type_text",
    "press_key",
    "hotkey",
    "move_mouse",
    "scroll",
    "drag",
    "focus_window",
    "clipboard_get",
    "clipboard_set",
    "clipboard_paste",
    "launch_application",
})

_ACTION_TYPE_MAP: dict[str, DesktopActionType] = {
    "click": DesktopActionType.CLICK,
    "double_click": DesktopActionType.DOUBLE_CLICK,
    "right_click": DesktopActionType.RIGHT_CLICK,
    "click_text_on_screen": DesktopActionType.CLICK_TEXT_ON_SCREEN,
    "click_screen_target": DesktopActionType.CLICK_SCREEN_TARGET,
    "double_click_screen_target": DesktopActionType.DOUBLE_CLICK_SCREEN_TARGET,
    "right_click_screen_target": DesktopActionType.RIGHT_CLICK_SCREEN_TARGET,
    "type_text": DesktopActionType.TYPE_TEXT,
    "press_key": DesktopActionType.PRESS_KEY,
    "hotkey": DesktopActionType.HOTKEY,
    "move_mouse": DesktopActionType.MOVE_MOUSE,
    "scroll": DesktopActionType.SCROLL,
    "drag": DesktopActionType.DRAG,
    "focus_window": DesktopActionType.FOCUS_WINDOW,
    "clipboard_get": DesktopActionType.CLIPBOARD_GET,
    "clipboard_set": DesktopActionType.CLIPBOARD_SET,
    "clipboard_paste": DesktopActionType.CLIPBOARD_PASTE,
    "launch_application": DesktopActionType.LAUNCH_APP,
}


def is_desktop_action(action_name: str) -> bool:
    """Return True if *action_name* should be handled by the desktop bridge."""
    return action_name in DESKTOP_TOOL_NAMES


def _build_desktop_action(
    action_name: str,
    params: dict[str, Any],
    *,
    description: str = "",
    expected_change: str = "",
) -> DesktopAction:
    """Convert an agent-loop step into a DesktopAction contract object."""
    action_type = _ACTION_TYPE_MAP.get(action_name)
    if action_type is None:
        raise ValueError(f"Unknown desktop action: '{action_name}'")

    return DesktopAction(
        action_type=action_type,
        params=dict(params),
        description=description or f"Agent step: {action_name}",
        expected_change=expected_change,
        metadata={"source": "agent_loop", "original_action": action_name},
    )


class DesktopBridge:
    """Runs desktop steps through observe→act→verify→recover.

    Wraps the DesktopMissionExecutor so the agent loop can call a single
    method and receive observation-enriched results.
    """

    def __init__(
        self,
        *,
        action_executor: DesktopActionExecutor | None = None,
        observer: DesktopObserver | None = None,
        mission_executor: DesktopMissionExecutor | None = None,
        approval_callback=None,
        max_retries: int = 1,
        min_confidence: float = 0.35,
        container: Any = None,
    ) -> None:
        self.action_executor = action_executor or (container.resolve("desktop_executor") if container else None) or DesktopActionExecutor()
        self.observer = observer or (container.resolve("desktop_observer") if container else None) or DesktopObserver()
        self.mission_executor = mission_executor or DesktopMissionExecutor(
            action_executor=self.action_executor,
            observer=self.observer,
            approval_callback=approval_callback,
            max_retries=max_retries,
            min_confidence=min_confidence,
        )

    async def execute_step(
        self,
        action_name: str,
        params: dict[str, Any],
        *,
        goal: str = "",
        description: str = "",
        expected_change: str = "",
    ) -> tuple[ToolObservation, MissionExecutionRecord]:
        """Run one desktop step through the full mission loop.

        Returns:
            (tool_observation, mission_record) — the observation is in the
            format the agent loop expects, and the record contains the full
            mission audit trail.
        """
        desktop_action = _build_desktop_action(
            action_name,
            params,
            description=description,
            expected_change=expected_change,
        )

        record = await self.mission_executor.run(
            goal=goal or f"Execute {action_name}",
            actions=[desktop_action],
            plan_summary=description,
        )

        # Convert MissionExecutionRecord into a ToolObservation for the agent loop.
        observation = _record_to_observation(action_name, record)
        return observation, record


def _record_to_observation(
    tool_name: str,
    record: MissionExecutionRecord,
) -> ToolObservation:
    """Convert a MissionExecutionRecord to a ToolObservation."""
    if record.status == DesktopMissionStatus.SUCCEEDED:
        execution_status = "success"
        output = record.explain()
        error = ""
    else:
        execution_status = "failure"
        output = ""
        error = record.explain()

    # Attach rich metadata from the mission record.
    metadata: dict[str, Any] = {
        "mission_id": record.mission_id,
        "mission_status": record.status.value,
        "duration_seconds": record.duration_seconds,
        "steps_completed": sum(1 for s in record.steps if s.status == "succeeded"),
        "steps_total": len(record.steps),
    }

    # Include observation data from the last step if available.
    if record.steps:
        last_step = record.steps[-1]
        action = last_step.action if isinstance(last_step.action, dict) else {}
        arguments = action.get("params", {}) if isinstance(action.get("params", {}), dict) else {}
        if last_step.change:
            metadata["change_detected"] = last_step.change.get("changed", False)
            metadata["change_summary"] = last_step.change.get("summary", "")
        if last_step.observation_after:
            metadata["final_confidence"] = last_step.observation_after.get("confidence", 0.0)
    else:
        arguments = {}

    return ToolObservation(
        tool_name=tool_name,
        arguments=arguments,
        execution_status=execution_status,
        output_summary=output,
        error_message=error,
        metadata=metadata,
    )


__all__ = [
    "DESKTOP_TOOL_NAMES",
    "DesktopBridge",
    "is_desktop_action",
]
