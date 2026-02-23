"""core/introspection/__init__.py"""

from core.introspection.explain import (
    explain_context,
    explain_last_actions,
    explain_policy_decision,
    explain_reflection,
    why_was_action_blocked,
)
from core.introspection.health import (
    HealthCheckResult,
    HealthReport,
    HealthStatus,
    run_health_check,
)
from core.introspection.state_dump import dump_state, pretty_dump

__all__ = [
    # explain
    "explain_context",
    "explain_last_actions",
    "explain_policy_decision",
    "explain_reflection",
    "why_was_action_blocked",
    # health
    "HealthCheckResult",
    "HealthReport",
    "HealthStatus",
    "run_health_check",
    # state dump
    "dump_state",
    "pretty_dump",
]
