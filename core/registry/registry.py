"""
Unified Capability Registry — replaces tool router and plugin manifest loading,
merging desktop and functional capabilities into a single dynamic registry.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import time
from pathlib import Path
from typing import Any, Callable

from core.autonomy.risk_evaluator import RiskLevel
from core.capability.base import Capability, ToolObservation, _normalize_tool_result
from core.context.context import TaskExecutionContext

# For desktop capability mapping
from core.desktop.contracts import DesktopAction, DesktopActionType
from core.desktop.mission import DesktopMissionExecutor, MissionExecutionRecord

logger = logging.getLogger("Jarvis.Registry")

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


def _build_desktop_action(
    action_name: str,
    params: dict[str, Any],
    *,
    description: str = "",
    expected_change: str = "",
    requires_approval: bool | None = None,
) -> DesktopAction:
    action_type = _ACTION_TYPE_MAP.get(action_name)
    if action_type is None:
        raise ValueError(f"Unknown desktop action: '{action_name}'")

    return DesktopAction(
        action_type=action_type,
        params=dict(params),
        description=description or f"Agent step: {action_name}",
        expected_change=expected_change,
        requires_approval=requires_approval,
        metadata={"source": "agent_loop", "original_action": action_name},
    )


def _record_to_observation(
    tool_name: str,
    record: MissionExecutionRecord,
) -> ToolObservation:
    # Handle both enum values and string statuses
    status_str = record.status.value if hasattr(record.status, "value") else str(record.status)
    if status_str.lower() in ("succeeded", "success"):
        execution_status = "success"
        output = record.explain()
        error = ""
    else:
        execution_status = "failure"
        output = ""
        error = record.explain()

    metadata: dict[str, Any] = {
        "mission_id": record.mission_id,
        "mission_status": status_str,
        "duration_seconds": record.duration_seconds,
        "steps_completed": sum(1 for s in record.steps if s.status == "succeeded"),
        "steps_total": len(record.steps),
    }

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


class FunctionCapability(Capability):
    """Adapts a standard python function to the Capability class interface."""

    def __init__(
        self,
        name: str,
        handler: Callable,
        risk_level: RiskLevel = RiskLevel.LOW,
        is_write: bool = False,
        description: str = "",
    ) -> None:
        self.name = name
        self.handler = handler
        self.risk_level = risk_level
        self.is_write = is_write
        self.description = description or (handler.__doc__ or "").strip()

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        sig = inspect.signature(self.handler)
        kwargs = dict(args)
        
        # Pass the task context if the handler accepts it
        if "context" in sig.parameters:
            kwargs["context"] = context

        # Support both coroutine handlers and synchronous blockers
        if inspect.iscoroutinefunction(self.handler):
            result = await self.handler(**kwargs)
        else:
            result = await asyncio.to_thread(self.handler, **kwargs)

        # Normalize outputs into ToolObservation properties
        success, output_summary, error_message = _normalize_tool_result(result)

        return ToolObservation(
            tool_name=self.name,
            arguments=args,
            execution_status="success" if success else "failure",
            output_summary=output_summary,
            error_message=error_message or None,
        )


class DesktopCapability(Capability):
    """Executes a desktop action through PyAutoGUI / Observe-Act-Verify loop."""

    def __init__(
        self,
        name: str,
        container: Any,
        is_write: bool = True,
        risk_level: RiskLevel = RiskLevel.CONFIRM,
    ) -> None:
        self.name = name
        self.container = container
        self.is_write = is_write
        self.risk_level = risk_level

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        # Resolve DesktopMissionExecutor from container on-demand to avoid boot cycles
        desktop_executor = self.container.resolve("desktop_executor") if self.container else None
        desktop_observer = self.container.resolve("desktop_observer") if self.container else None
        
        mission_executor = DesktopMissionExecutor(
            action_executor=desktop_executor,
            observer=desktop_observer,
            max_retries=1,
            min_confidence=0.35,
        )

        requires_approval = None
        if context.get("approval_called") and context.get("approval_result"):
            requires_approval = False

        desktop_action = _build_desktop_action(
            self.name,
            args,
            description=args.get("description", f"Desktop command: {self.name}"),
            expected_change=args.get("expected_change", ""),
            requires_approval=requires_approval,
        )

        record = await mission_executor.run(
            goal=context.variables.get("goal", f"Execute {self.name}"),
            actions=[desktop_action],
            plan_summary=args.get("description", ""),
        )

        return _record_to_observation(self.name, record)


class CapabilityRegistry:
    """Unified Registry for local capabilities, API tools, and dynamically loaded plugins."""

    def __init__(self, container: Any = None) -> None:
        self.container = container
        self._capabilities: dict[str, Capability] = {}
        self._call_count = 0
        self._observations: list[ToolObservation] = []

    def register(self, name_or_cap: str | Capability, handler: Callable | None = None) -> None:
        """Register a tool, accepting either a Capability subclass instance or legacy name/handler."""
        if isinstance(name_or_cap, Capability):
            name = name_or_cap.name.strip().lower()
            self._capabilities[name] = name_or_cap
            logger.debug(f"Registered Capability: {name}")
            return

        if handler is None:
            raise ValueError("Handler must be provided for function registration.")

        name = name_or_cap.strip().lower()
        
        # Self-declare properties for functional tools
        is_write = name not in {
            "get_time", "get_system_stats", "list_directory", "read_file",
            "search_memory", "capture_screen", "capture_region",
            "find_text_on_screen", "read_screen_text", "wait_for_text_on_screen",
            "describe_screen", "get_active_window", "clipboard_get",
            "web_search", "web_scrape", "list_hardware_devices",
            "ping_device", "read_sensor",
        }
        
        risk_level = RiskLevel.LOW
        if name in {
            "shell_exec", "shell", "exec", "subprocess", "delete_file", "rm", "rmdir"
        }:
            risk_level = RiskLevel.CRITICAL
        elif name in DESKTOP_TOOL_NAMES or is_write:
            risk_level = RiskLevel.CONFIRM

        if name in DESKTOP_TOOL_NAMES:
            self._capabilities[name] = DesktopCapability(name, container=self.container, is_write=is_write, risk_level=risk_level)
        else:
            self._capabilities[name] = FunctionCapability(name, handler, risk_level=risk_level, is_write=is_write)
        logger.debug(f"Adapted function registry: {name}")

    def get(self, name: str) -> Capability | None:
        return self._capabilities.get(name.strip().lower())

    def registered_tools(self) -> list[str]:
        return list(self._capabilities.keys())

    def reset_call_count(self) -> None:
        self._call_count = 0

    async def execute(self, tool_name: str, arguments: dict, context: TaskExecutionContext | None = None) -> ToolObservation:
        if context is None:
            context = TaskExecutionContext()

        cap = self.get(tool_name)
        if not cap:
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=f"No capability registered for tool '{tool_name}'.",
            )
            self._observations.append(obs)
            return obs

        logger.info(f"[CAPABILITY LOG] Executing: {tool_name}({arguments})", extra={"trace_id": context.trace_id, "task_id": context.task_id})
        self._call_count += 1
        start = time.monotonic()

        try:
            obs = await cap.run(arguments, context)
            obs.duration_seconds = time.monotonic() - start
            if obs.execution_status == "success":
                logger.info(f"[CAPABILITY OK] {tool_name} completed.", extra={"trace_id": context.trace_id, "task_id": context.task_id})
            else:
                logger.warning(f"[CAPABILITY FAIL] {tool_name} failed: {obs.error_message}", extra={"trace_id": context.trace_id, "task_id": context.task_id})
        except Exception as e:
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=str(e),
                duration_seconds=time.monotonic() - start,
            )
            logger.error(f"[CAPABILITY ERROR] {tool_name}: {e}", exc_info=True, extra={"trace_id": context.trace_id, "task_id": context.task_id})

        self._observations.append(obs)
        if len(self._observations) > 1000:
            self._observations = self._observations[-500:]
        return obs

    def get_observations(self) -> list[ToolObservation]:
        return list(self._observations)

    def clear_observations(self) -> None:
        self._observations.clear()

    def load_plugins(self, plugin_dir: str | Path) -> list[str]:
        directory = Path(plugin_dir)
        if not directory.exists() or not directory.is_dir():
            return []

        loaded: list[str] = []
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"jarvis_plugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                register_fn = getattr(module, "register", None)
                if callable(register_fn):
                    register_fn(self)
                    loaded.append(path.stem)
            except Exception:
                continue
        return loaded
