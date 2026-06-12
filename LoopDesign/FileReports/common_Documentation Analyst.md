# Analysis Report for common.py

## Dependencies
- __future__.annotations
- dataclasses.dataclass
- dataclasses.field
- enum.Enum
- typing.Any

## Schemas
- ToolResult
- ToolResult attribute: success
- ToolResult attribute: data
- ToolResult attribute: error
- ToolResult attribute: tool_name
- IntegrationRiskLevel

## API Contracts
- ToolResult.to_llm_string(self)
- ToolResult.__repr__(self)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Common type definitions for Jarvis.

