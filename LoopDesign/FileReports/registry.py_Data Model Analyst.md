# File Report: registry.py
**Path**: `d:\AI\Jarvis\integrations\registry.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- inspect
- logging
- typing.Any
- integrations.base.BaseIntegration

## Classes and State Objects
### `IntegrationRegistry`
**Variables**: None
**Methods**: __init__, register, register_safety_rules, get_tools, list_schemas, get_tool, list_tools, execute, list_active

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for name, integration in self._integrations.items():
            tools = integration.get_tools() or []
            for tool in tools:
                if isinstance(tool, dict):
                    tool_name = tool.get("name")
                    if tool_name and self._tool_owner.get(tool_name) == name:
                        merged.append(dict(tool))
        return merged

```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.