# Documentation Report: registry.py

## Assumptions
- Maintains mapping of `_integrations` (name -> integration) and `_tool_owner` (tool_name -> integration_name).
- `register()` replaces existing tools ownership seamlessly.
- Tool dictionary must declare `"name"`. Tool skipping occurs if name is empty.
- Risk string from tool definition (`"risk": "low" | "medium" | "high" | "read_only"`) determines the safety classification. `"low"` and variations of `"read-only"` map to read-only tools in the AutonomyGovernor, others map to write tools.
- `execute` extracts keyword or positional arguments dynamically based on the tool's execute function signature. Coerces output to `{"success": bool, "data": Any, "error": str}`.
- Exceptions raised in `execute` are caught and wrapped.

## Schema / API Contract
- `register(integration: BaseIntegration)`
- `register_safety_rules(autonomy_governor, risk_evaluator)` scans tools and applies rules.
- `get_tools() -> list[dict[str, Any]]` returns flattened tool schemas.
- `execute(tool_name: str, args: dict[str, Any]) -> dict[str, Any]`

## Dependencies
- `inspect`, `logging`, `typing`
- `integrations.base.BaseIntegration`

## Configuration Variables
None.

## Prompts
None.
