# File Report: registry.py
## Role: Dependency Analyst

### 1. Library Requirements
- `inspect` (Standard Library)
- `logging` (Standard Library)
- `typing` (Standard Library)
- `integrations.base` (Local): imports `BaseIntegration`

### 2. Service Dependencies
- None.

### 3. Hidden Execution Links
- `execute()` function dynamically resolves a tool name to an integration and calls its `execute` method using `getattr(integration, "execute")`.
- Awaits the result if the execute function is a coroutine (`inspect.isawaitable`).
- Links tools to external modules like `AutonomyGovernor` and `RiskEvaluator` based on the defined `"risk"` level.

### 4. Assumptions & API Contracts
- `tool_name` uniqueness: Tools are registered by name. If two plugins define the same tool name, the latest registered plugin silently overwrites ownership.
- The tools returned by `get_tools()` from integrations are expected to be dictionaries containing `name` and `risk`.
- The `execute` method of integrations can either use `**kwargs` or accept a single `args` dictionary, handled dynamically via signature inspection.
- Normalizes returned payload from tools. If an integration doesn't return a dict, attempts to parse `success`, `data`, and `error` attributes.

### 5. Configuration Variables
- Risk levels in schemas dictate security constraints:
  - Low/Read-Only: `"low"`, `"read_only"`, `"read-only"` mapped to read operations.
  - Medium: `"medium"`.
  - High/Write: Anything else (default, `"confirm"`, `"high"`).

### 6. Prompts Found
- None.
