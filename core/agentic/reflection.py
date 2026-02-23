# core/agentic/reflection.py
# Reflection / post-action analysis module

from typing import Optional


def reflect(context: Optional[dict] = None) -> dict:
    """
    Perform self-reflection on the agent's last action or decision.

    Args:
        context: Optional dict containing state, action, outcome, etc.

    Returns:
        A structured reflection result dict.
    """
    if context is None:
        context = {}

    try:
        action = context.get("action", "unknown")
        outcome = context.get("outcome", "unknown")
        confidence = context.get("confidence", 0.0)

        lessons = []

        if outcome == "failure":
            lessons.append("Action did not produce expected result.")
        elif outcome == "success":
            lessons.append("Action completed successfully.")
        else:
            lessons.append("Outcome undetermined; further observation needed.")

        return {
            "action": action,
            "outcome": outcome,
            "confidence": confidence,
            "lessons": lessons,
            "status": "reflected",
        }

    except Exception as e:
        return {
            "action": None,
            "outcome": None,
            "confidence": 0.0,
            "lessons": [],
            "status": "error",
            "error": str(e),
        }
