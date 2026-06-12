# File Report: script_ast.py
**Path**: `d:\AI\Jarvis\integrations\script_ast.py`
**Role**: Data Model Analyst

## Analysis Summary
This file has been analyzed for schemas, DTOs, state objects, config variables, and dependencies.

## Dependencies
- os
- ast
- json
- re

## Classes and State Objects
### `Analyzer`
**Variables**: None
**Methods**: __init__, visit_Import, visit_ImportFrom, visit_Call, visit_Subscript, visit_Assign, visit_Dict, visit_ClassDef, visit_FunctionDef

## Tool Schemas / DTOs
No explicit tool schemas found.

## Assumptions & API Contracts
1. Config vars are expected in environment variables.
2. Schema validation is typically deferred to the registry or client implementation.