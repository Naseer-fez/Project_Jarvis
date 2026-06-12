# Analysis Report for shortcuts.py

## Dependencies
- __future__.annotations
- re
- dataclasses.dataclass
- pathlib.Path
- urllib.parse.quote_plus
- core.desktop.actions.DesktopActionExecutor
- core.desktop.contracts.DesktopAction
- core.desktop.contracts.DesktopActionType
- core.desktop.mission.DesktopMissionExecutor
- core.desktop.mission.DesktopMissionStatus
- core.desktop.observation.DesktopObserver
- core.tools.system_automation.async_launch_application

## Schemas
- DesktopCommandPlan
- DesktopCommandPlan attribute: app_label
- DesktopCommandPlan attribute: primary_target
- DesktopCommandPlan attribute: primary_args
- DesktopCommandPlan attribute: response_text

## API Contracts
- _supported_apps_message()
- _extract_search_query(lowered, original)
- plan_desktop_command(user_input)

## Configuration Variables
- INTERACTIVE_VERBS_PATTERN
- PROJECT_ROOT

## Assumptions & Notes
- Module Docstring: Simple, safe desktop shortcuts for common open/search commands.

