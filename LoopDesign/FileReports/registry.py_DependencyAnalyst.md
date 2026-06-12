# Dependency Analysis Report for registry\registry.py

## Library Requirements
- from __future__ import annotations
- from core.autonomy.risk_evaluator import RiskLevel
- from core.capability.base import Capability
- from core.capability.base import ToolObservation
- from core.capability.base import _normalize_tool_result
- from core.context.context import TaskExecutionContext
- from core.desktop.contracts import DesktopAction
- from core.desktop.contracts import DesktopActionType
- from core.desktop.mission import DesktopMissionExecutor
- from core.desktop.mission import MissionExecutionRecord
- from pathlib import Path
- from typing import Any
- from typing import Callable
- import asyncio
- import importlib.util
- import inspect
- import logging
- import time

## Service Dependencies
- asyncio.to_thread

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
