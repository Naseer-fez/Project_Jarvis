"""
Plan schema utilities for Jarvis planner output.

The planner may return partial JSON. This module normalizes plans into a
stable shape required by the execution layer.
"""

from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = (
    "intent",
    "steps",
    "tools_required",
    "risk_level",
    "confirmation_required",
)

ALLOWED_RISK_LEVELS = {"low", "medium", "high", "critical"}


def build_unknown_plan(intent: str, reason: str = "") -> dict[str, Any]:
    prompt = (
        reason
        if reason
        else "I don't have a safe way to complete this request. Please rephrase."
    )
    return {
        "intent": intent,
        "summary": "I don't know how to do that safely.",
        "confidence": 0.0,
        "steps": [],
        "tools_required": [],
        "risk_level": "high",
        "confirmation_required": True,
        "clarification_needed": True,
        "clarification_prompt": prompt,
    }


def infer_tools_required(steps: list[dict[str, Any]]) -> list[str]:
    tools: list[str] = []
    seen: set[str] = set()
    for step in steps:
        action = str(step.get("action", "")).strip().lower()
        if action and action not in seen:
            seen.add(action)
            tools.append(action)
    return tools


def _normalize_steps(raw_steps: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, list):
        return []

    steps: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            continue
        params = raw.get("params", {})
        if not isinstance(params, dict):
            params = {}
        steps.append(
            {
                "id": int(raw.get("id", idx)),
                "action": str(raw.get("action", "")).strip().lower(),
                "description": str(raw.get("description", "")).strip(),
                "params": params,
            }
        )
    return steps


def normalize_plan(plan: dict[str, Any], intent: str) -> dict[str, Any]:
    """
    Normalize any planner output into the required execution schema.
    Never raises.
    """
    plan = plan if isinstance(plan, dict) else {}
    steps = _normalize_steps(plan.get("steps", []))
    tools_required = plan.get("tools_required", [])
    if not isinstance(tools_required, list) or not tools_required:
        tools_required = infer_tools_required(steps)
    else:
        deduped: list[str] = []
        seen: set[str] = set()
        for tool in tools_required:
            t = str(tool).strip().lower()
            if t and t not in seen:
                seen.add(t)
                deduped.append(t)
        tools_required = deduped

    risk_level = str(plan.get("risk_level", "low")).strip().lower()
    if risk_level not in ALLOWED_RISK_LEVELS:
        risk_level = "low"

    return {
        "intent": str(intent),
        "summary": str(plan.get("summary", "No summary.")),
        "confidence": float(plan.get("confidence", 0.0)),
        "steps": steps,
        "tools_required": tools_required,
        "risk_level": risk_level,
        "confirmation_required": bool(plan.get("confirmation_required", False)),
        "clarification_needed": bool(plan.get("clarification_needed", False)),
        "clarification_prompt": str(plan.get("clarification_prompt", "")),
    }
