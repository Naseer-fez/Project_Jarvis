"""
core/registry/base.py
─────────────────────
Abstract base class for all dynamically loadable tools and capabilities in Jarvis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from core.context.context import TaskExecutionContext


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    CONFIRM = "confirm"
    HIGH = "high"
    CRITICAL = "critical"


class Capability(ABC):
    """Abstract base class for all tools and integrations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the capability."""
        pass

    @property
    @abstractmethod
    def is_write_operation(self) -> bool:
        """True if the tool mutates state, files, or sends outbound payloads."""
        pass

    @property
    @abstractmethod
    def risk_level(self) -> RiskLevel:
        """The tool risk profile (e.g. LOW, CRITICAL)."""
        pass

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """JSON schema defining the expected arguments."""
        pass

    @abstractmethod
    async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> dict[str, Any]:
        """Asynchronous, non-blocking execution callback."""
        pass
