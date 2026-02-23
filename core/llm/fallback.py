# core/llm/fallback.py
# LLM failure fallback handler — ALWAYS returns a valid plan object.

from typing import Optional


def fallback_plan(prompt: Optional[str] = None, reason: Optional[str] = None) -> dict:
    """
    Return a safe fallback plan when the primary LLM planner fails.

    Args:
        prompt: The original prompt that triggered the failure (optional).
        reason: Human-readable reason for the fallback (optional).

    Returns:
        A valid fallback plan dict. Never returns None.
    """
    try:
        return {
            "plan": [],
            "confidence": 0.0,
            "reason": reason or "fallback",
            "prompt": prompt,
            "status": "fallback",
        }
    except Exception:
        # Last-resort return — must never propagate an exception
        return {
            "plan": [],
            "confidence": 0.0,
            "reason": "fallback",
            "prompt": None,
            "status": "fallback",
        }


def handle_planner_failure(error: Optional[Exception] = None) -> dict:
    """
    Convenience wrapper called on planner crash or timeout.

    Args:
        error: The exception that caused the failure (optional).

    Returns:
        A valid fallback plan dict.
    """
    reason = str(error) if error else "planner failure"
    return fallback_plan(reason=reason)


def handle_schema_error(schema_name: Optional[str] = None) -> dict:
    """
    Convenience wrapper called on invalid schema output.

    Args:
        schema_name: Name of the schema that failed validation (optional).

    Returns:
        A valid fallback plan dict.
    """
    reason = f"invalid schema: {schema_name}" if schema_name else "invalid schema"
    return fallback_plan(reason=reason)
