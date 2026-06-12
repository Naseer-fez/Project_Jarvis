# File Report: planner.py
**Role**: Prompt Recovery Specialist

## Dependencies
- inspect
- core.autonomy.risk_evaluator
- re
- typing
- json
- logging
- __future__

## Configuration Variables & Constants
- `risk_label`: `critical`
- `risk_label`: `high`
- `risk_label`: `medium`
- `risk_label`: `low`

## Schemas & API Contracts
### Class `TaskPlanner`
**Methods**: __init__, _tool_schema, _call_ollama, plan, _build_prompt, _parse_llm_plan, _fallback_plan, _clarification_plan, _enrich_plan, _normalize_steps

### Function `_strip_planner_artifacts`
**Args**: raw

### Function `__init__`
**Args**: self, config, llm, registry

### Function `_tool_schema`
**Args**: self

### Function `_call_ollama`
**Args**: self, prompt

### Function `plan`
**Args**: self, user_input, context

### Function `_build_prompt`
**Args**: self, user_input, context

### Function `_parse_llm_plan`
**Args**: self, raw

### Function `_fallback_plan`
**Args**: self, text

### Function `_clarification_plan`
**Args**: self, text

### Function `_enrich_plan`
**Args**: self, text, plan

### Function `_normalize_steps`
**Args**: self, steps

## Prompts and LLM Directives
No explicit prompts found in module scope.
