# `base.py` - API Analyst Report

## Overview
Defines the `BaseIntegration` abstract base class which serves as the contract for all dynamic integrations within Jarvis.

## Endpoints / Tools
No concrete tools implemented here, but defines the interface:
- `is_available() -> bool`: Determines if dependencies and configurations are satisfied.
- `get_tools() -> list[dict[str, Any]]`: Returns a list of tool schema dictionaries in the planner `SYSTEM_TOOL_SCHEMA` format.
- `execute(tool_name: str, args: dict[str, Any]) -> IntegrationResult`: Executes a tool and returns a normalized payload containing `{"success": bool, "data": Any, "error": str | None}`.

## External Contracts / Dependencies
- Relies on `core.types.common` types (`IntegrationResult`, `ToolResult`, `IntegrationRiskLevel`, `RiskLevel`).
- The output of `execute` uses a unified schema.

## Assumptions
- Subclasses must implement `is_available`, `get_tools`, and `execute`.
- Integration tools will have a standardized argument dictionary (`args`).
- `unavailable_reason` must be used to track missing dependencies.
- Expected to fail silently on unavailability during initialization.
