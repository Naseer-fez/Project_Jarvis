# File Report: agent_loop.py
**Role**: Prompt Recovery Specialist

## Dependencies
- inspect
- core.planner.planner
- httpx
- core.metrics.confidence
- time
- core.autonomy.risk_evaluator
- re
- core.autonomy.autonomy_governor
- typing
- logging
- dataclasses
- asyncio
- core.context.context
- core.registry.registry
- __future__
- core.state_machine
- sys
- traceback

## Configuration Variables & Constants
- `REFLECT_SYSTEM_PROMPT`: (Too long, 281 chars. Extracted to Prompts if applicable)
- `user_prompt`: `f-string: Goal:
{...}

Plan:
{...}

Tool observations:
{...}
`
- `obs_text`: `No tool observations.`

## Schemas & API Contracts
### Class `ExecutionTrace`
**Methods**: close, to_dict

### Class `AgentLoopEngine`
**Methods**: __init__, request_interrupt, _check_interrupt, run, _ensure_thinking_state, _build_plan, _normalize_steps, _ask_confirmation, _reflect, _plan_summary, _fallback_reflection, _stop

### Function `_truncate_obs`
**Args**: text, max_chars
**Assumptions/Doc**: Truncate long observations to keep both leading and trailing context.

### Function `_truncate_observation`
**Args**: text, max_chars

### Function `close`
**Args**: self, success, reason

### Function `to_dict`
**Args**: self

### Function `__init__`
**Args**: self, state_machine, task_planner, tool_router, risk_evaluator, autonomy_governor, model, ollama_url, max_iterations, llm, container

### Function `request_interrupt`
**Args**: self

### Function `_check_interrupt`
**Args**: self

### Function `run`
**Args**: self, goal, context, confirm_callback

### Function `_ensure_thinking_state`
**Args**: self, sm

### Function `_build_plan`
**Args**: self, goal, context

### Function `_normalize_steps`
**Args**: self, plan

### Function `_ask_confirmation`
**Args**: self, prompt, confirm_callback, context

### Function `_reflect`
**Args**: self, goal, plan, observations, trace

### Function `_plan_summary`
**Args**: self, plan

### Function `_fallback_reflection`
**Args**: self, plan, observations

### Function `_stop`
**Args**: self, trace, reason, sm

## Prompts and LLM Directives
- Extracted `REFLECT_SYSTEM_PROMPT` to Prompts directory.
- Extracted `user_prompt` to Prompts directory.
