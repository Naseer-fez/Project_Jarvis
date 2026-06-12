# registry.py API Analyst Report

## Overview
Registry and execution router that maps active tool names to their integration owners and handles routing tool executions.

## API Contracts & Methods
- `IntegrationRegistry`
  - `register(integration: BaseIntegration)`: Registers an integration and its tools.
  - `register_safety_rules(autonomy_governor: Any, risk_evaluator: Any)`: Extracts `risk` level from tool schemas and registers them appropriately.
    - Low/Read-only: `autonomy_governor.register_read_only_tool`, `risk_evaluator.register_low_action`
    - Medium: write action, `risk_evaluator.register_medium_action`
    - High/Confirm: write action, `risk_evaluator.register_confirm_action`
  - `get_tools() -> list[dict[str, Any]]`: Returns all registered tool schemas.
  - `list_schemas() -> list[dict[str, Any]]`: Alias for `get_tools()`.
  - `get_tool(tool_name: str) -> BaseIntegration | None`
  - `list_tools() -> dict[str, str]`: Maps tool name -> integration name.
  - `execute(tool_name: str, args: dict[str, Any]) -> dict[str, Any]`: Async method that dynamically routes execution to the integration owner. Normalizes output.

## Assumptions
- Tool schemas must contain `name` and optional `risk` string keys.
- Normalizes execution results to `{"success": bool, "data": Any, "error": str}`.

## Prompts
- None.
