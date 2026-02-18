"""
ToolRouter — dispatches tool invocations to concrete implementations.
All calls are logged before execution. Sandbox enforcement is applied.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("Jarvis.ToolRouter")

TOOL_TIMEOUT_SECONDS = 15
MAX_TOOL_CALLS_PER_GOAL = 10

ToolHandler = Callable[..., Coroutine[Any, Any, Any]]


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
            result = await asyncio.wait_for(
                handler(**arguments),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
            duration = time.monotonic() - start
            obs = ToolObservation(
                tool_name=tool_name,
                arguments=arguments,
                execution_status="success",
                output_summary=str(result)[:1000],
                duration_seconds=duration,
            )
            logger.info(f"[TOOL OK] {tool_name} → {obs.output_summary[:120]}")
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

