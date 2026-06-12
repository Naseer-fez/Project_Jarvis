# File Report: loader.py
**Path**: `d:\AI\Jarvis\integrations\loader.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- __future__.annotations
- importlib
- inspect
- logging
- pathlib.Path
- typing.Any
- integrations.base.BaseIntegration

## Classes and State Objects
### `IntegrationLoader`
**Variables**: None
**Methods**: load_all

## Tool Schemas / DTOs
No explicit tool schemas found.

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.