# core/llm/v1_flow_schema_condition.py
# Schema conditions for the v1 planner flow.
# Static definitions only — no execution on import.

from typing import Any


# ---------------------------------------------------------------------------
# Condition type constants
# ---------------------------------------------------------------------------

CONDITION_TYPE_EQUALS = "equals"
CONDITION_TYPE_CONTAINS = "contains"
CONDITION_TYPE_EXISTS = "exists"
CONDITION_TYPE_NOT_EXISTS = "not_exists"
CONDITION_TYPE_GT = "greater_than"
CONDITION_TYPE_LT = "less_than"

SUPPORTED_CONDITION_TYPES = [
    CONDITION_TYPE_EQUALS,
    CONDITION_TYPE_CONTAINS,
    CONDITION_TYPE_EXISTS,
    CONDITION_TYPE_NOT_EXISTS,
    CONDITION_TYPE_GT,
    CONDITION_TYPE_LT,
]


# ---------------------------------------------------------------------------
# Base schema shapes (plain dicts, no runtime evaluation)
# ---------------------------------------------------------------------------

BASE_CONDITION_SCHEMA: dict = {
    "type": "object",
    "required": ["condition_type", "field"],
    "properties": {
        "condition_type": {
            "type": "string",
            "enum": SUPPORTED_CONDITION_TYPES,
        },
        "field": {"type": "string"},
        "value": {"type": "any"},
        "negate": {"type": "boolean", "default": False},
    },
}

FLOW_CONDITION_SCHEMA: dict = {
    "version": "v1",
    "schema": BASE_CONDITION_SCHEMA,
    "allow_nested": True,
    "max_depth": 3,
}


# ---------------------------------------------------------------------------
# Placeholder condition presets
# ---------------------------------------------------------------------------

CONDITION_PRESETS: dict = {
    "always_true": {
        "condition_type": CONDITION_TYPE_EXISTS,
        "field": "__always__",
        "negate": False,
    },
    "always_false": {
        "condition_type": CONDITION_TYPE_EXISTS,
        "field": "__never__",
        "negate": True,
    },
    "has_plan": {
        "condition_type": CONDITION_TYPE_EXISTS,
        "field": "plan",
        "negate": False,
    },
    "plan_empty": {
        "condition_type": CONDITION_TYPE_EQUALS,
        "field": "plan",
        "value": [],
        "negate": False,
    },
    "confidence_high": {
        "condition_type": CONDITION_TYPE_GT,
        "field": "confidence",
        "value": 0.7,
        "negate": False,
    },
}


def get_preset(name: str) -> dict:
    """
    Return a condition preset by name, or an empty dict if not found.

    Args:
        name: Preset key from CONDITION_PRESETS.

    Returns:
        Condition dict or empty dict.
    """
    return dict(CONDITION_PRESETS.get(name, {}))


def get_flow_schema() -> dict:
    """Return a copy of the v1 flow condition schema."""
    return dict(FLOW_CONDITION_SCHEMA)
