# Analysis Report for planner.py

## Dependencies
- __future__.annotations
- json
- logging
- re
- inspect
- typing.Any
- core.autonomy.risk_evaluator.RiskLevel
- core.autonomy.risk_evaluator.RiskEvaluator

## Schemas
- TaskPlanner

## API Contracts
- _strip_planner_artifacts(raw)
- TaskPlanner.__init__(self, config, llm, registry)
- TaskPlanner._tool_schema(self)
- TaskPlanner._build_prompt(self, user_input, context)
- TaskPlanner._parse_llm_plan(self, raw)
- TaskPlanner._fallback_plan(self, text)
- TaskPlanner._clarification_plan(self, text)
- TaskPlanner._enrich_plan(self, text, plan)
- TaskPlanner._normalize_steps(self, steps)

## Configuration Variables
- _GUI_TOOL_NAMES

## Assumptions & Notes
- Module Docstring: Asynchronous task planner to generate execution plans.

