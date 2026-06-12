# Documentation Report: base.py

## Assumptions
- Every integration inherits from `BaseIntegration`
- `is_available` returns boolean natively without raising exceptions; failure reasons are populated in `self.unavailable_reason`
- Tools are declared in `get_tools()` returning a list of dicts mapping to `SYSTEM_TOOL_SCHEMA` format
- `execute` is asynchronous and returns an `IntegrationResult` compatible dictionary: `{"success": bool, "data": Any, "error": str | None}`

## Schema / API Contract
- `IntegrationResult`, `ToolResult`, `IntegrationRiskLevel`, `RiskLevel` are imported from `core.types.common`
- `name` (str), `description` (str), `required_config` (list[str]) are class variables defining the integration.
- Initialization accepts an optional `config` object.

## Dependencies
- `abc` (stdlib)
- `typing` (stdlib)
- `core.types.common` (internal)

## Configuration Variables
- No direct env vars, configuration is assumed to be passed during init or handled by children.

## Prompts
None.
