"""
core/risk_evaluator.py — Compatibility shim.

Legacy test files import:
    from core.risk_evaluator import RiskEvaluator, RiskLevel

The canonical implementation lives in core/autonomy/risk_evaluator.py.
"""

from core.autonomy.risk_evaluator import RiskEvaluator, RiskLevel, RiskResult  # noqa: F401

__all__ = ["RiskEvaluator", "RiskLevel", "RiskResult"]
