# Analysis Report for risk_evaluator.py

## Dependencies
- __future__.annotations
- threading
- dataclasses.dataclass
- dataclasses.field
- enum.IntEnum
- typing.Sequence
- typing.Any

## Schemas
- RiskLevel
- RiskResult
- RiskResult attribute: level
- RiskResult attribute: blocking_actions
- RiskResult attribute: confirm_actions
- RiskResult attribute: high_risk_actions
- RiskResult attribute: reasons
- RiskEvaluator

## API Contracts
- RiskLevel.label(self)
- RiskResult.is_blocked(self)
- RiskResult.requires_confirmation(self)
- RiskResult.summary(self)
- RiskEvaluator.__init__(self, config, registry)
- RiskEvaluator.register_critical_action(self, action)
- RiskEvaluator.register_confirm_action(self, action)
- RiskEvaluator.register_high_action(self, action)
- RiskEvaluator.register_medium_action(self, action)
- RiskEvaluator.register_low_action(self, action)
- RiskEvaluator._load_config(self, config)
- RiskEvaluator.evaluate(self, actions)
- RiskEvaluator.evaluate_plan(self, plan)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Deterministic risk evaluator for planned tool actions without hardcoded tool strings.

