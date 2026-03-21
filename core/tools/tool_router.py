"""
ToolRouter — dispatches tool invocations to concrete implementations.
All calls are logged before execution. Sandbox enforcement is applied.
"""

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("Jarvis.ToolRouter")

TOOL_TIMEOUT_SECONDS = 15
MAX_TOOL_CALLS_PER_GOAL = 10

ToolHandler = Callable[..., Any]


@dataclass
class ToolObservation:
    tool_name: str
    arguments: dict
    execution_status: str       # "success" | "failure"
    output_summary: str
    error_message: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "execution_status": self.execution_status,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "duration_seconds": round(self.duration_seconds, 3),
        }


class ToolRouter:
    def __init__(self):
        self._registry: dict[str, ToolHandler] = {}
        self._call_count = 0
        self._observations: list[ToolObservation] = []

    def register(self, name: str, handler: ToolHandler):
        self._registry[name] = handler
        logger.debug(f"Registered tool: {name}")

    def registered_tools(self) -> list[str]:
        return list(self._registry.keys())

    def reset_call_count(self):
        self._call_count = 0

    async def execute(self, tool_name: str, arguments: dict) -> ToolObservation:
        if self._call_count >= MAX_TOOL_CALLS_PER_GOAL:
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=f"Max tool calls per goal ({MAX_TOOL_CALLS_PER_GOAL}) exceeded.",
            )
            self._observations.append(obs)
            return obs

        handler = self._registry.get(tool_name)
        if not handler:
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=f"No handler registered for tool '{tool_name}'.",
            )
            self._observations.append(obs)
            return obs

        logger.info(f"[TOOL LOG] Executing: {tool_name}({arguments})")
        self._call_count += 1
        start = time.monotonic()

        try:
            if inspect.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**arguments),
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(handler, **arguments),
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
            duration = time.monotonic() - start
            success, output_summary, error_message = _normalize_tool_result(result)
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="success" if success else "failure",
                output_summary=output_summary[:1000],
                error_message=(error_message or None),
                duration_seconds=duration,
            )
            if success:
                logger.info(f"[TOOL OK] {tool_name} → {obs.output_summary[:120]}")
            else:
                logger.warning(f"[TOOL FAIL] {tool_name} → {obs.error_message}")
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT_SECONDS}s.",
                duration_seconds=duration,
            )
            logger.error(f"[TOOL TIMEOUT] {tool_name}")
        except Exception as e:
            duration = time.monotonic() - start
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="failure",
                output_summary="",
                error_message=str(e),
                duration_seconds=duration,
            )
            logger.error(f"[TOOL ERROR] {tool_name}: {e}")

        self._observations.append(obs)
        return obs

    def get_observations(self) -> list[ToolObservation]:
        return list(self._observations)

    def clear_observations(self):
        self._observations.clear()


def _normalize_tool_result(result: Any) -> tuple[bool, str, str]:
    if isinstance(result, dict) and "success" in result:
        success = bool(result.get("success", False))
        output = _first_non_empty(
            result.get("output"),
            result.get("data"),
            result.get("metadata"),
        )
        error = str(result.get("error", "") or "")
        if success:
            return True, output or "Tool completed successfully.", ""
        return False, output, error or "Tool returned an error."

    success_attr = getattr(result, "success", None)
    if success_attr is None:
        text = _stringify_payload(result)
        return True, text or "Tool completed successfully.", ""

    success = bool(success_attr)
    output = _first_non_empty(
        getattr(result, "output", None),
        getattr(result, "data", None),
        getattr(result, "metadata", None),
    )
    error = str(getattr(result, "error", "") or "")
    if success:
        return True, output or "Tool completed successfully.", ""
    return False, output, error or _stringify_payload(result) or "Tool returned an error."


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _stringify_payload(value)
        if text:
            return text
    return ""


def _stringify_payload(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return str(value)

