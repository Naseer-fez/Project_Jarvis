"""
core/planning/plan_schema.py
Shim that re-exports from core.llm.plan_schema for backward compatibility.
"""
from core.llm.plan_schema import build_unknown_plan, normalize_plan

__all__ = ["build_unknown_plan", "normalize_plan"]
