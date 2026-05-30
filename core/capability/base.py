"""
Capability — Base class for all tools in Jarvis.
"""

from __future__ import annotations

from typing import Any
from core.autonomy.risk_evaluator import RiskLevel
from core.tools.tool_router import ToolObservation
from core.context.context import TaskExecutionContext


class Capability:
    """Base class for all tools and capabilities in Jarvis."""

    name: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    is_write: bool = False

    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation:
        """Execute the capability logic in the provided task context."""
        raise NotImplementedError
