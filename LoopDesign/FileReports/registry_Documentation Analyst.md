# Analysis Report for registry.py

## Dependencies
- __future__.annotations
- asyncio
- importlib.util
- inspect
- logging
- time
- pathlib.Path
- typing.Any
- typing.Callable
- core.autonomy.risk_evaluator.RiskLevel
- core.capability.base.Capability
- core.capability.base.ToolObservation
- core.capability.base._normalize_tool_result
- core.context.context.TaskExecutionContext
- core.desktop.contracts.DesktopAction
- core.desktop.contracts.DesktopActionType
- core.desktop.mission.DesktopMissionExecutor
- core.desktop.mission.MissionExecutionRecord

## Schemas
- FunctionCapability
- DesktopCapability
- CapabilityRegistry

## API Contracts
- _build_desktop_action(action_name, params)
- _record_to_observation(tool_name, record)
- FunctionCapability.__init__(self, name, handler, risk_level, is_write, description)
- DesktopCapability.__init__(self, name, container, is_write, risk_level)
- CapabilityRegistry.__init__(self, container)
- CapabilityRegistry.register(self, name_or_cap, handler)
- CapabilityRegistry.get(self, name)
- CapabilityRegistry.registered_tools(self)
- CapabilityRegistry.reset_call_count(self)
- CapabilityRegistry.get_observations(self)
- CapabilityRegistry.clear_observations(self)
- CapabilityRegistry.load_plugins(self, plugin_dir)

## Configuration Variables
- DESKTOP_TOOL_NAMES (typed)
- _ACTION_TYPE_MAP (typed)

## Assumptions & Notes
- Module Docstring: Unified Capability Registry — replaces tool router and plugin manifest loading,
merging desktop and functional capabilities into a single dynamic registry.

