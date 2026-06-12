# base.py API Analyst Report

## Overview
Defines the `BaseIntegration` abstract base class which establishes the contract for all dynamic integrations in Jarvis.

## API Contracts & Methods
- `BaseIntegration(ABC)`
  - Attributes: 
    - `name: str`
    - `description: str`
    - `required_config: list[str]`
  - `__init__(config)`: Takes an optional configuration object.
  - `is_available() -> bool`: Abstract method to check if dependencies/env vars are present.
  - `get_tools() -> list[dict[str, Any]]`: Abstract method to return tool schemas in planner SYSTEM_TOOL_SCHEMA format.
  - `execute(tool_name: str, args: dict[str, Any]) -> IntegrationResult`: Abstract async method to execute a tool.

## Schemas
- `IntegrationResult`: Implicitly expected to match `{"success": bool, "data": Any, "error": str | None}` as normalized payload.

## Dependencies
- `core.types.common` (IntegrationResult, ToolResult, IntegrationRiskLevel, RiskLevel)

## Prompts
- None.
