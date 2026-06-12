# File Report: base.py
## Role: Dependency Analyst

### 1. Library Requirements
- `abc` (Standard Library)
- `typing` (Standard Library)
- `core.types.common` (Local internal module): imports `IntegrationResult`, `ToolResult`, `IntegrationRiskLevel`, `RiskLevel`

### 2. Service Dependencies
- None directly. Defines the abstract structure for integration plugins.

### 3. Hidden Execution Links
- Acts as a required superclass for all dynamically loaded integrations.
- `IntegrationLoader` depends on inspecting modules for subclasses of `BaseIntegration`.

### 4. Assumptions & API Contracts
- `is_available()`: Must return a boolean. Must *not* raise exceptions. If False, the integration is silently skipped.
- `get_tools()`: Must return a list of dictionaries adhering to the planner's `SYSTEM_TOOL_SCHEMA`.
- `execute()`: Must be an async function accepting `tool_name: str` and `args: dict[str, Any]`. Must return a normalized payload `IntegrationResult` dictionary: `{"success": bool, "data": Any, "error": str | None}`.
- Attributes `name`, `description`, and `required_config` are expected to be defined on subclasses.

### 5. Configuration Variables
- `self.config`: Generic property to store configurations.
- `unavailable_reason`: Optional string set by subclass to explain why it is disabled.
- `required_config`: List of strings indicating required environment variables (contracted, but not enforced in this base class).

### 6. Prompts Found
- None.
