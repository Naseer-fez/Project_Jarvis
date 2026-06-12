# Analysis Report for base.py

## Dependencies
- __future__.annotations
- abc.ABC
- abc.abstractmethod
- enum.Enum
- typing.Any
- core.capability.base.ToolObservation
- core.context.context.TaskExecutionContext

## Schemas
- RiskLevel
- Capability

## API Contracts
- Capability.name(self)
- Capability.is_write_operation(self)
- Capability.risk_level(self)
- Capability.schema(self)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: core/registry/base.py
─────────────────────
Abstract base class for all dynamically loadable tools and capabilities in Jarvis.

