"""Routing helpers for dispatcher tool classification and risk lookup."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping


def is_desktop_action(tool_name: str, desktop_map: Mapping[str, object]) -> bool:
    return tool_name in desktop_map


def is_core_tool(tool_name: str, core_tools: Mapping[str, object]) -> bool:
    return tool_name in core_tools


def is_integration_tool(
    tool_name: str,
    integration_checker: Callable[[str], bool],
) -> bool:
    return integration_checker(tool_name)


def core_or_desktop_risk_score(tool_name: str, registry: Mapping[str, float]) -> float:
    return float(registry.get(tool_name, 1.0))


def integration_risk_score(tool_name: str, registry: Mapping[str, float]) -> float:
    return float(registry.get(tool_name, 0.6))


__all__ = [
    "core_or_desktop_risk_score",
    "integration_risk_score",
    "is_core_tool",
    "is_desktop_action",
    "is_integration_tool",
]
