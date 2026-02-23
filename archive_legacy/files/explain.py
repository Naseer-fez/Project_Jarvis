"""
core/introspection/explain.py

Explainability interface — answers "why did Jarvis do X?" in natural language.

Sources of truth:
- DecisionTrace (what was decided and why)
- ReflectionRecord history (what the agent learned)
- AutonomyPolicy audit log (what was allowed / blocked)
- AgentContext (what the state was at the time)
"""

from __future__ import annotations

from typing import Any, Optional


def explain_last_actions(decision_trace: Any, n: int = 5) -> str:
    """
    Return a plain-English explanation of the last N decisions.

    Args:
        decision_trace: DecisionTrace instance.
        n: Number of recent entries to include.
    """
    if decision_trace is None:
        return "No decision trace available."
    return decision_trace.explain_last(n)


def explain_policy_decision(policy_decision: Any) -> str:
    """
    Return a plain-English explanation of a single policy decision.

    Args:
        policy_decision: PolicyDecision instance.
    """
    verdict = policy_decision.verdict.value.upper()
    return (
        f"[Policy: {verdict}] Action '{policy_decision.action_name}' "
        f"was evaluated by rule '{policy_decision.rule_name}'. "
        f"Reason: {policy_decision.reason}"
    )


def explain_reflection(reflection_record: Any) -> str:
    """
    Return a plain-English explanation of a reflection outcome.

    Args:
        reflection_record: ReflectionRecord instance.
    """
    r = reflection_record
    conf_delta = r.confidence_after - r.confidence_before
    risk_delta = r.risk_after - r.risk_before
    delta_str = (
        f"Confidence {'+' if conf_delta >= 0 else ''}{conf_delta:.2f}, "
        f"Risk {'+' if risk_delta >= 0 else ''}{risk_delta:.2f}."
    )
    lessons_str = "\n  - ".join(r.lessons) if r.lessons else "(none)"
    return (
        f"Reflection [{r.verdict.value}]: {r.summary}\n"
        f"Score changes: {delta_str}\n"
        f"Lessons:\n  - {lessons_str}"
    )


def explain_context(context: Any) -> str:
    """
    Return a plain-English summary of the current AgentContext.

    Args:
        context: AgentContext instance.
    """
    lines = [
        f"Session: {context.session_id}",
        f"Current goal: {context.current_goal_id or 'none'}",
        f"Current mission: {context.mission_id or 'none'}",
        f"Confidence: {context.confidence_score:.2f}",
        f"Risk: {context.risk_score:.2f}",
        f"Safe to proceed: {context.is_safe_to_proceed}",
        f"Interrupt: {context.interrupt_flag}",
        f"Paused: {context.paused}",
    ]
    return "\n".join(lines)


def why_was_action_blocked(
    action_name: str,
    policy: Any,
    override_protocol: Optional[Any] = None,
) -> str:
    """
    Look up in the policy audit log and override log to explain a block.

    Args:
        action_name:       The action that was blocked.
        policy:            AutonomyPolicy instance.
        override_protocol: Optional HumanOverrideProtocol.
    """
    explanations: list[str] = []

    # Check policy audit log
    decisions = policy.audit_log(action_name)
    denials = [d for d in decisions if d.verdict.value in ("deny", "require_approval")]
    for d in denials[-3:]:
        explanations.append(f"Policy rule '{d.rule_name}': {d.reason}")

    # Check override log
    if override_protocol is not None:
        overrides = override_protocol.log_for(action_name)
        for o in overrides[-3:]:
            if o.override_type.value in ("deny", "permanent_ban", "emergency_stop"):
                explanations.append(
                    f"Human override ({o.override_type.value}) by '{o.issued_by}': {o.reason}"
                )

    if not explanations:
        return f"No recorded blocks found for action '{action_name}'."

    header = f"Action '{action_name}' was blocked for the following reason(s):"
    return header + "\n" + "\n".join(f"  • {e}" for e in explanations)

