"""
core/agentic/autonomy_policy.py

Hard safety rules that decide whether the agent may act autonomously
on a given action.  Nothing in this file is heuristic — every rule is
explicit and auditable by a human reviewer.

Responsibilities:
- Gate every action through policy checks
- Return a structured PolicyDecision (go / no-go / require-approval)
- Never perform the action itself
- Log every decision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from core.agentic.agent_context import AgentContext


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PolicyVerdict(str, Enum):
    ALLOW           = "allow"            # proceed autonomously
    REQUIRE_APPROVAL = "require_approval" # pause and ask human
    DENY            = "deny"             # hard block — do not proceed


@dataclass(frozen=True)
class PolicyDecision:
    """Immutable result of a policy check."""

    verdict: PolicyVerdict
    reason: str
    rule_name: str
    action_name: str
    context_snapshot: dict = field(default_factory=dict)
    decided_at: datetime = field(default_factory=_utcnow)

    @property
    def allowed(self) -> bool:
        return self.verdict == PolicyVerdict.ALLOW

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "rule_name": self.rule_name,
            "action_name": self.action_name,
            "context_snapshot": self.context_snapshot,
            "decided_at": self.decided_at.isoformat(),
        }


# ── Rule type ────────────────────────────────────────────────────────────────

RuleFunc = Callable[["AgentContext", str, dict], Optional[PolicyDecision]]
"""
A rule is a callable:
    (context, action_name, action_params) -> PolicyDecision | None

Return None to abstain (no opinion); return a PolicyDecision to rule.
Rules are evaluated in order; first non-None result wins.
"""


# ── Built-in rules ───────────────────────────────────────────────────────────

def _rule_interrupt_flag(ctx: "AgentContext", action: str, params: dict) -> Optional[PolicyDecision]:
    if ctx.interrupt_flag:
        return PolicyDecision(
            verdict=PolicyVerdict.DENY,
            reason="Interrupt flag is raised — agent is halted.",
            rule_name="interrupt_flag",
            action_name=action,
            context_snapshot=ctx.snapshot(),
        )
    return None


def _rule_risk_threshold(ctx: "AgentContext", action: str, params: dict) -> Optional[PolicyDecision]:
    if ctx.risk_score > 0.85:
        return PolicyDecision(
            verdict=PolicyVerdict.DENY,
            reason=f"Risk score {ctx.risk_score:.2f} exceeds hard limit 0.85.",
            rule_name="risk_threshold_hard",
            action_name=action,
            context_snapshot=ctx.snapshot(),
        )
    if ctx.risk_score > 0.6:
        return PolicyDecision(
            verdict=PolicyVerdict.REQUIRE_APPROVAL,
            reason=f"Risk score {ctx.risk_score:.2f} exceeds soft limit 0.60.",
            rule_name="risk_threshold_soft",
            action_name=action,
            context_snapshot=ctx.snapshot(),
        )
    return None


def _rule_low_confidence(ctx: "AgentContext", action: str, params: dict) -> Optional[PolicyDecision]:
    if ctx.confidence_score < 0.3:
        return PolicyDecision(
            verdict=PolicyVerdict.REQUIRE_APPROVAL,
            reason=f"Confidence {ctx.confidence_score:.2f} is too low to act autonomously.",
            rule_name="low_confidence",
            action_name=action,
            context_snapshot=ctx.snapshot(),
        )
    return None


def _rule_paused(ctx: "AgentContext", action: str, params: dict) -> Optional[PolicyDecision]:
    if ctx.paused:
        return PolicyDecision(
            verdict=PolicyVerdict.DENY,
            reason="Agent is paused — awaiting human input.",
            rule_name="agent_paused",
            action_name=action,
            context_snapshot=ctx.snapshot(),
        )
    return None


def _rule_dangerous_tools(ctx: "AgentContext", action: str, params: dict) -> Optional[PolicyDecision]:
    """Hard-block a short list of obviously dangerous tool names."""
    BLOCKED = {
        "delete_all_files",
        "format_disk",
        "drop_database",
        "send_mass_email",
        "execute_arbitrary_code",
    }
    if action in BLOCKED:
        return PolicyDecision(
            verdict=PolicyVerdict.DENY,
            reason=f"Action '{action}' is on the permanent blocklist.",
            rule_name="dangerous_tools_blocklist",
            action_name=action,
            context_snapshot=ctx.snapshot(),
        )
    return None


_DEFAULT_RULES: list[RuleFunc] = [
    _rule_interrupt_flag,
    _rule_paused,
    _rule_dangerous_tools,
    _rule_risk_threshold,
    _rule_low_confidence,
]


# ── Policy engine ────────────────────────────────────────────────────────────

class AutonomyPolicy:
    """
    Evaluates a proposed action against an ordered list of rules.

    Usage:
        policy = AutonomyPolicy()
        decision = policy.check(context, action_name="send_email", params={...})
        if not decision.allowed:
            handle_denial(decision)
    """

    def __init__(self, extra_rules: Optional[list[RuleFunc]] = None) -> None:
        self._rules: list[RuleFunc] = list(_DEFAULT_RULES)
        if extra_rules:
            self._rules.extend(extra_rules)
        self._audit_log: list[PolicyDecision] = []

    def add_rule(self, rule: RuleFunc, prepend: bool = False) -> None:
        """Add a custom rule. Prepend=True makes it highest priority."""
        if prepend:
            self._rules.insert(0, rule)
        else:
            self._rules.append(rule)

    def check(
        self,
        context: "AgentContext",
        action_name: str,
        params: Optional[dict] = None,
    ) -> PolicyDecision:
        """
        Run all rules and return the first non-None decision.
        If all rules abstain, ALLOW by default.
        """
        params = params or {}
        for rule in self._rules:
            decision = rule(context, action_name, params)
            if decision is not None:
                self._audit_log.append(decision)
                return decision

        # All rules abstained → allow
        allow = PolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            reason="All policy rules passed.",
            rule_name="default_allow",
            action_name=action_name,
            context_snapshot=context.snapshot(),
        )
        self._audit_log.append(allow)
        return allow

    def audit_log(self, action_name: Optional[str] = None) -> list[PolicyDecision]:
        if action_name:
            return [d for d in self._audit_log if d.action_name == action_name]
        return list(self._audit_log)
