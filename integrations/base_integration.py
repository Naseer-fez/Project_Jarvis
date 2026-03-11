"""Legacy integration helpers kept for backward compatibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import BaseIntegration


class RiskLevel(str, Enum):
    READ_ONLY = "READ_ONLY_TOOLS"
    CONFIRM = "CONFIRM_TOOLS"
    HIGH_RISK = "HIGH_RISK_TOOLS"


@dataclass
class ToolResult:
    """Compatibility result object for legacy integration modules."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    tool_name: str = ""

    def to_llm_string(self) -> str:
        if self.success:
            return f"{self.tool_name or 'tool'} success: {self.data}"
        return f"{self.tool_name or 'tool'} error: {self.error}"


__all__ = ["BaseIntegration", "RiskLevel", "ToolResult"]
