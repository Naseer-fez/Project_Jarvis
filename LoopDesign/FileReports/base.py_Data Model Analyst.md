# File Report: base.py
**Path**: `d:\AI\Jarvis\integrations\base.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- abc.ABC
- typing.Any
- core.types.common.IntegrationResult

## Classes and State Objects
### `BaseIntegration`
**Variables**: None
**Methods**: __init__, is_available, get_tools, execute

## Tool Schemas / DTOs
```python
    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool schema dicts in the planner SYSTEM_TOOL_SCHEMA format."""

    @abstractmethod
```

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.