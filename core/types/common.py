"""Common type definitions for Jarvis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


IntegrationResult = dict[str, Any]

@dataclass
class ToolResult:
    """Standardised return type for all Jarvis tool functions.

    Attributes:
        success: True if the tool call succeeded.
        data:    Payload on success (arbitrary dict).
        error:   Human-readable error message on failure.
        tool_name: (Optional) Name of the tool.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""

    def to_llm_string(self) -> str:
        if self.success:
            return f"{self.tool_name or 'tool'} success: {self.data}"
        return f"{self.tool_name or 'tool'} error: {self.error}"

    def __repr__(self) -> str:
        if self.success:
            return f"ToolResult(success=True, data={self.data})"
        return f"ToolResult(success=False, error={self.error!r})"


class IntegrationRiskLevel(str, Enum):
    READ_ONLY = "READ_ONLY_TOOLS"
    CONFIRM = "CONFIRM_TOOLS"
    HIGH_RISK = "HIGH_RISK_TOOLS"


# Backward compatibility alias
RiskLevel = IntegrationRiskLevel
