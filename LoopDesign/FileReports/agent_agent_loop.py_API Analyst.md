# API Analyst Report: agent\agent_loop.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import inspect`
- `import logging`
- `import re`
- `import time`
- `from dataclasses import dataclass`
- `from dataclasses import field`
- `from typing import Any`
- `from typing import Optional`
- `from core.state_machine import State as AgentState`
- `from core.state_machine import StateMachine`
- `from core.context.context import TaskExecutionContext`
- `from core.autonomy.autonomy_governor import AutonomyGovernor`
- `from core.autonomy.risk_evaluator import RiskEvaluator`
- `from core.planner.planner import TaskPlanner`
- `from core.metrics.confidence import ConfidenceModel`
- `from core.registry.registry import ToolObservation`
- `from core.registry.registry import CapabilityRegistry`

## Configuration Variables
- `_DEFAULT_MAX_ITERATIONS` = `10`

## Prompts Extracted

- `REFLECT_SYSTEM_PROMPT` -> Saved to `Prompts/agent_loop_REFLECT_SYSTEM_PROMPT.txt`

## Schemas & API Contracts (Classes)

### Class `ExecutionTrace`
**Fields/Schema:**
  - `goal: str`
  - `iterations: int`
  - `plan: Optional[dict[str, Any]]`
  - `observations: list[dict[str, Any]]`
  - `risk_scores: list[dict[str, Any]]`
  - `think_blocks: list[str]`
  - `reflection: Optional[str]`
  - `final_response: str`
  - `success: bool`
  - `stop_reason: str`
  - `started_at: float`
  - `ended_at: Optional[float]`

**Methods:**
- `def close(self, success: bool, reason: str) -> None`
- `def to_dict(self) -> dict[str, Any]`


### Class `AgentLoopEngine`
**Methods:**
- `def __init__(self, state_machine: StateMachine | None=None, task_planner: TaskPlanner | None=None, tool_router: CapabilityRegistry | None=None, risk_evaluator: RiskEvaluator | None=None, autonomy_governor: AutonomyGovernor | None=None, model: str='mistral', ollama_url: str='http://localhost:11434', max_iterations: int=_DEFAULT_MAX_ITERATIONS, llm: Any=None, container: Any=None)`
- `def request_interrupt(self) -> None`
- `def _check_interrupt(self) -> bool`
- `async def run(self, goal: str, context: TaskExecutionContext, confirm_callback=None) -> ExecutionTrace`
- `def _ensure_thinking_state(self, sm: StateMachine) -> None`
- `async def _build_plan(self, goal: str, context: str) -> dict[str, Any]`
- `def _normalize_steps(self, plan: dict[str, Any]) -> list[dict[str, Any]]`
- `async def _ask_confirmation(self, prompt: str, confirm_callback, context: TaskExecutionContext) -> bool`
- `async def _reflect(self, goal: str, plan: dict[str, Any], observations: list[ToolObservation], trace: ExecutionTrace) -> str`
- `def _plan_summary(self, plan: dict[str, Any]) -> str`
- `def _fallback_reflection(self, plan: dict[str, Any], observations: list[ToolObservation]) -> str`
- `def _stop(self, trace: ExecutionTrace, reason: str, sm: StateMachine) -> ExecutionTrace`


## Functions & Endpoints

### `_truncate_obs`
`def _truncate_obs(text: str, max_chars: int=800) -> str`
> Truncate long observations to keep both leading and trailing context.

### `_truncate_observation`
`def _truncate_observation(text: str, max_chars: int=800) -> str`