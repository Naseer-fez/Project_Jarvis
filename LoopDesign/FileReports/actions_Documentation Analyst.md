# Analysis Report for actions.py

## Dependencies
- __future__.annotations
- inspect
- time
- typing.Any
- typing.Callable
- core.autonomy.risk_evaluator.RiskEvaluator
- core.autonomy.risk_evaluator.RiskLevel
- core.autonomy.risk_evaluator.RiskResult
- core.desktop.contracts.DesktopAction
- core.desktop.contracts.DesktopActionResult
- core.desktop.contracts.DesktopActionStatus
- core.desktop.contracts.DesktopActionType
- core.desktop.contracts.DesktopRiskTier

## Schemas
- DesktopActionExecutor

## API Contracts
- _stringify(value)
- _normalize_tool_result(result)
- DesktopActionExecutor.__init__(self)
- DesktopActionExecutor.evaluate_risk(self, action)
- DesktopActionExecutor.requires_approval(self, action)
- DesktopActionExecutor._audit(self, action, result)
- DesktopActionExecutor._contains_sensitive_text(action)
- DesktopActionExecutor._result(action)
- DesktopActionExecutor._default_handlers()

## Configuration Variables
- _SENSITIVE_TEXT_MARKERS

## Assumptions & Notes
- Module Docstring: Normalized desktop action execution with risk and audit metadata.

