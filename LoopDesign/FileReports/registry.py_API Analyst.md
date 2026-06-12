# `registry.py` - API Analyst Report

## Overview
Serves as the registry and execution router for active Jarvis integrations. Stores loaded tools and maps them to their owner integrations.

## Endpoints / Tools
- `register(integration)`: Registers integration tools, replacing any stale tool ownership maps.
- `register_safety_rules(autonomy_governor, risk_evaluator)`: Introspects tool schemas to register read/write and risk rules with the agent's safety systems.
- `execute(tool_name: str, args: dict[str, Any])`: Looks up the integration responsible for `tool_name` and runs its `execute` method. Awaits the result if asynchronous. 

## External Contracts / Dependencies
- Works intimately with `BaseIntegration`.
- `AutonomyGovernor` and `RiskEvaluator` interact with this registry through `register_safety_rules`. Tools marked with `risk: "low" | "read_only" | "read-only"` are registered as safe/read tools, and others are write/confirm tools.
- Output from executions are normalized into standard dictionary formats: `{"success": bool, "data": Any, "error": str}`.

## Assumptions
- Integration names must not be empty.
- A single integration tool schema format is required. Tool names must be uniquely identifiable (though warnings are logged if a tool name is reassigned, indicating possible collisions).
- Exposes `integration_registry` and an alias `api_registry`.
