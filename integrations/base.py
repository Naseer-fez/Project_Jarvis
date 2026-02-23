"""
integrations/base_integration.py

Abstract base class for all Jarvis external API integrations.
All integrations MUST inherit from this class and implement its interface.
This ensures every plugin is compatible with tool_router.py and the AutonomyGovernor.
"""

from __future__ import annotations

import abc
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Risk classification — must align with AutonomyGovernor permission levels
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Maps directly to AutonomyGovernor permission levels."""
    READ_ONLY = "READ_ONLY_TOOLS"   # Level 2 — data fetch, no side-effects
    WRITE     = "WRITE_TOOLS"       # Level 3 — mutating, sends, purchases, etc.


# ---------------------------------------------------------------------------
# Standard tool result envelope
# ---------------------------------------------------------------------------

class ToolResult:
    """
    Uniform return type for every integration action.
    The dispatcher / tool_router should always receive one of these,
    never a raw exception or bare dict.
    """

    def __init__(
        self,
        *,
        success: bool,
        data: Any = None,
        error: str | None = None,
        tool_name: str = "",
    ) -> None:
        self.success   = success
        self.data      = data
        self.error     = error
        self.tool_name = tool_name

    def to_llm_string(self) -> str:
        """
        Serialise result for injection into the LLM prompt.
        Returns a clean string in either case so the planner never crashes.
        """
        if self.success:
            return str(self.data)
        return f"{{'error': '{self.error}'}}"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ToolResult(success={self.success}, tool={self.tool_name!r}, "
            f"error={self.error!r})"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseIntegration(abc.ABC):
    """
    Every external API integration must subclass this.

    Contract
    --------
    * ``tool_name``   — unique snake_case identifier registered in api_registry
    * ``risk_level``  — READ_ONLY or WRITE; used by AutonomyGovernor.can_execute()
    * ``tool_schema`` — JSON schema dict exposed to task_planner / LLM
    * ``execute()``   — performs the actual API call; MUST return ToolResult
                        and MUST never raise — catch all exceptions internally.
    """

    # Subclasses MUST set these at class level (or override as properties)
    tool_name:  str       = NotImplemented
    risk_level: RiskLevel = NotImplemented

    # ------------------------------------------------------------------
    # Schema — override in subclass (or import from tool_schema.py)
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def tool_schema(self) -> dict:
        """
        Return a JSON-serialisable dict describing how the planner should call
        this tool.  Minimum required keys: name, description, parameters.
        """

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Perform the API action.

        Rules
        -----
        * Catch ALL exceptions — return ToolResult(success=False, error=...) on failure.
        * On network / timeout errors return the offline-degradation error string.
        * Never modify core/  or memory/  state directly.
        * Log via core.logger BEFORE and AFTER the external call.
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _offline_result(self) -> ToolResult:
        """Standard offline-degradation response."""
        return ToolResult(
            success=False,
            error="Offline mode active, cannot reach API",
            tool_name=self.tool_name,
        )

    def _error_result(self, exc: Exception) -> ToolResult:
        """Wrap any unexpected exception into a clean ToolResult."""
        return ToolResult(
            success=False,
            error=f"Integration error [{self.tool_name}]: {type(exc).__name__}: {exc}",
            tool_name=self.tool_name,
        )

