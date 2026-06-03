"""
Capability — Base class for all tools in Jarvis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from core.autonomy.risk_evaluator import RiskLevel
from core.context.context import TaskExecutionContext

logger = logging.getLogger("Jarvis.Capability")




class Capability:
    """Base class for all tools and capabilities in Jarvis."""

    name: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    is_write: bool = False

    @property
    def is_write_operation(self) -> bool:
        """Alias for is_write, conforming to the abstract Capability base interface."""
        return self.is_write

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        """Execute the capability logic in the provided task context."""
        raise NotImplementedError


@dataclass
class ToolObservation:
    tool_name: str
    arguments: dict
    execution_status: str       # "success" | "failure"
    output_summary: str
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "execution_status": self.execution_status,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": dict(self.metadata or {}),
        }





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

