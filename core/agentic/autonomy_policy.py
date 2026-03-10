"""Autonomy policy decisions for the Agentic Layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .belief_state import BeliefState

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_risk(value: float) -> float:
    risk = float(value)
    if risk < 0.0:
        return 0.0
    if risk > 1.0:
        return 1.0
    return round(risk, 4)


class PolicyVerdict(str, Enum):
    """Policy verdicts exposed to the rest of the system."""

    ALLOW = "allow"
    REQUIRE_CONFIRM = "require_approval"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    """Structured output for a policy evaluation."""

    verdict: PolicyVerdict
    reason: str
    action_name: str
    rule_name: str
    risk_score: float = 0.0
    confidence_score: float = 0.0
    decided_at: datetime = field(default_factory=_utcnow)
    metadata: dict[str, object] = field(default_factory=dict)

    def is_allowed(self) -> bool:
        return self.verdict == PolicyVerdict.ALLOW

    def requires_confirmation(self) -> bool:
        return self.verdict == PolicyVerdict.REQUIRE_CONFIRM

    def is_denied(self) -> bool:
        return self.verdict == PolicyVerdict.DENY

    @property
    def allowed(self) -> bool:
        return self.is_allowed()

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "action_name": self.action_name,
            "rule_name": self.rule_name,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "decided_at": self.decided_at.isoformat(),
            "metadata": dict(self.metadata),
        }


class AutonomyPolicy:
    """Policy engine driven by belief state and explicit risk thresholds."""

    HARD_DENY_ACTIONS = {
        "disable_logging",
        "bypass_auth",
        "self_modify_policy",
    }

    ALWAYS_CONFIRM_ACTIONS = {
        "send_email",
        "delete_file",
        "deploy_to_production",
    }

    def __init__(self, belief_state: BeliefState) -> None:
        self.belief_state = belief_state
        self._retry_counts: dict[str, int] = {}
        self._audit_log: list[PolicyDecision] = []

    def _record_decision(
        self,
        verdict: PolicyVerdict,
        action_name: str,
        reason: str,
        rule_name: str,
        risk_score: float,
        metadata: dict[str, object] | None = None,
    ) -> PolicyDecision:
        decision = PolicyDecision(
            verdict=verdict,
            reason=reason,
            action_name=action_name,
            rule_name=rule_name,
            risk_score=_clamp_risk(risk_score),
            confidence_score=self.belief_state.agent_confidence,
            metadata=dict(metadata or {}),
        )
        self._audit_log.append(decision)
        logger.info(
            "Policy decision action=%s verdict=%s rule=%s risk=%.3f reason=%s",
            action_name,
            decision.verdict.name,
            rule_name,
            decision.risk_score,
            reason,
        )
        return decision

    def evaluate(self, action_key: str, risk_score: float = 0.0) -> PolicyDecision:
        """Evaluate a single action using explicit policy rules."""
        normalized_action = str(action_key).strip().lower()
        normalized_risk = _clamp_risk(risk_score)

        if normalized_action in self.HARD_DENY_ACTIONS:
            return self._record_decision(
                PolicyVerdict.DENY,
                normalized_action,
                "Action is on the hard-deny list.",
                "hard_deny_list",
                normalized_risk,
            )

        if normalized_risk >= 0.95:
            return self._record_decision(
                PolicyVerdict.DENY,
                normalized_action,
                f"Risk score {normalized_risk:.2f} exceeds the deny threshold.",
                "risk_threshold_deny",
                normalized_risk,
            )

        if normalized_action in self.ALWAYS_CONFIRM_ACTIONS:
            return self._record_decision(
                PolicyVerdict.REQUIRE_CONFIRM,
                normalized_action,
                "Action is on the always-confirm list.",
                "always_confirm_list",
                normalized_risk,
            )

        if normalized_risk >= 0.6:
            return self._record_decision(
                PolicyVerdict.REQUIRE_CONFIRM,
                normalized_action,
                f"Risk score {normalized_risk:.2f} exceeds the confirm threshold.",
                "risk_threshold_confirm",
                normalized_risk,
            )

        if self.belief_state.agent_confidence < 0.4:
            return self._record_decision(
                PolicyVerdict.REQUIRE_CONFIRM,
                normalized_action,
                "Agent confidence is too low for autonomous execution.",
                "low_agent_confidence",
                normalized_risk,
            )

        if self.belief_state.risk_tolerance < 0.35 and normalized_risk > 0.2:
            return self._record_decision(
                PolicyVerdict.REQUIRE_CONFIRM,
                normalized_action,
                "Current risk tolerance is too low for this action.",
                "low_risk_tolerance",
                normalized_risk,
            )

        if self.belief_state.should_ask_user():
            return self._record_decision(
                PolicyVerdict.REQUIRE_CONFIRM,
                normalized_action,
                "Belief state indicates the user should confirm before proceeding.",
                "belief_state_confirmation",
                normalized_risk,
            )

        return self._record_decision(
            PolicyVerdict.ALLOW,
            normalized_action,
            "Action passed all autonomy policy checks.",
            "default_allow",
            normalized_risk,
        )

    def check(
        self,
        context: object,
        action_name: str,
        params: dict[str, object] | None = None,
    ) -> PolicyDecision:
        """Compatibility wrapper used by existing dispatcher code."""
        if getattr(context, "interrupt_flag", False):
            return self._record_decision(
                PolicyVerdict.DENY,
                action_name,
                "Interrupt flag is raised.",
                "context_interrupt_flag",
                float(getattr(context, "risk_score", 0.0)),
            )

        if getattr(context, "paused", False):
            return self._record_decision(
                PolicyVerdict.DENY,
                action_name,
                "Execution context is paused.",
                "context_paused",
                float(getattr(context, "risk_score", 0.0)),
            )

        risk_score = 0.0
        if params and "risk_score" in params:
            risk_score = float(params["risk_score"])
        elif hasattr(context, "risk_score"):
            risk_score = float(getattr(context, "risk_score"))

        decision = self.evaluate(action_name, risk_score=risk_score)

        if (
            decision.verdict == PolicyVerdict.ALLOW
            and hasattr(context, "confidence_score")
            and float(getattr(context, "confidence_score")) < 0.35
        ):
            return self._record_decision(
                PolicyVerdict.REQUIRE_CONFIRM,
                action_name,
                "Context confidence is too low for autonomous execution.",
                "context_low_confidence",
                risk_score,
            )

        return decision

    def check_retry(self, action_key: str, max_retries: int = 3) -> bool:
        """Increment and validate a retry counter for an action."""
        normalized_action = str(action_key).strip().lower()
        current_retries = self._retry_counts.get(normalized_action, 0)
        if current_retries >= max(0, int(max_retries)):
            logger.warning(
                "Retry denied for action=%s current=%s max=%s",
                normalized_action,
                current_retries,
                max_retries,
            )
            return False

        self._retry_counts[normalized_action] = current_retries + 1
        logger.info(
            "Retry allowed for action=%s attempt=%s max=%s",
            normalized_action,
            self._retry_counts[normalized_action],
            max_retries,
        )
        return True

    def reset_retry(self, action_key: str) -> None:
        normalized_action = str(action_key).strip().lower()
        if normalized_action in self._retry_counts:
            del self._retry_counts[normalized_action]

    def should_escalate(self, stall_count: int) -> bool:
        """Escalate after repeated stalls or low confidence conditions."""
        threshold = 1 if self.belief_state.should_ask_user() else 2
        if self.belief_state.agent_confidence < 0.3:
            threshold = 1
        should_raise = int(stall_count) >= threshold
        logger.info(
            "Escalation check stalls=%s threshold=%s result=%s",
            stall_count,
            threshold,
            should_raise,
        )
        return should_raise

    def audit_log(self, action_name: str | None = None) -> list[PolicyDecision]:
        if action_name is None:
            return list(self._audit_log)
        normalized_action = str(action_name).strip().lower()
        return [entry for entry in self._audit_log if entry.action_name == normalized_action]


__all__ = ["AutonomyPolicy", "PolicyDecision", "PolicyVerdict"]
