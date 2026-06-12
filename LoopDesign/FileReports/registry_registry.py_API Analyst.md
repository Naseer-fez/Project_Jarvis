# API Analyst Report: registry\registry.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import importlib.util`
- `import inspect`
- `import logging`
- `import time`
- `from pathlib import Path`
- `from typing import Any`
- `from typing import Callable`
- `from core.autonomy.risk_evaluator import RiskLevel`
- `from core.capability.base import Capability`
- `from core.capability.base import ToolObservation`
- `from core.capability.base import _normalize_tool_result`
- `from core.context.context import TaskExecutionContext`
- `from core.desktop.contracts import DesktopAction`
- `from core.desktop.contracts import DesktopActionType`
- `from core.desktop.mission import DesktopMissionExecutor`
- `from core.desktop.mission import MissionExecutionRecord`

## Schemas & API Contracts (Classes)

### Class `FunctionCapability(Capability)`
> Adapts a standard python function to the Capability class interface.

**Methods:**
- `def __init__(self, name: str, handler: Callable, risk_level: RiskLevel=RiskLevel.LOW, is_write: bool=False, description: str='') -> None`
- `async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation`


### Class `DesktopCapability(Capability)`
> Executes a desktop action through PyAutoGUI / Observe-Act-Verify loop.

**Methods:**
- `def __init__(self, name: str, container: Any, is_write: bool=True, risk_level: RiskLevel=RiskLevel.CONFIRM) -> None`
- `async def run(self, args: dict[str, Any], context: TaskExecutionContext) -> ToolObservation`


### Class `CapabilityRegistry`
> Unified Registry for local capabilities, API tools, and dynamically loaded plugins.

**Methods:**
- `def __init__(self, container: Any=None) -> None`
- `def register(self, name_or_cap: str | Capability, handler: Callable | None=None) -> None`
  - *Register a tool, accepting either a Capability subclass instance or legacy name/handler.*
- `def get(self, name: str) -> Capability | None`
- `def registered_tools(self) -> list[str]`
- `def reset_call_count(self) -> None`
- `async def execute(self, tool_name: str, arguments: dict, context: TaskExecutionContext | None=None) -> ToolObservation`
- `def get_observations(self) -> list[ToolObservation]`
- `def clear_observations(self) -> None`
- `def load_plugins(self, plugin_dir: str | Path) -> list[str]`


## Functions & Endpoints

### `_build_desktop_action`
`def _build_desktop_action(action_name: str, params: dict[str, Any], *, description: str='', expected_change: str='', requires_approval: bool | None=None) -> DesktopAction`
### `_record_to_observation`
`def _record_to_observation(tool_name: str, record: MissionExecutionRecord) -> ToolObservation`