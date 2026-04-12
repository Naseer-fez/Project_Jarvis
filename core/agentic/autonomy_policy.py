"""Decision policy for whether an agentic action can run automatically."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PolicyVerdict(str, Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRM = "require_confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    verdict: PolicyVerdict
    reason: str
    rule_name: str


class AutonomyPolicy:
    HARD_DENY = {"disable_logging", "delete_audit_log", "wipe_memory"}
    ALWAYS_CONFIRM = {
        "send_email",
        "send_whatsapp",
        "send_telegram",
        "delete_file",
        "execute_shell",
        "click",
        "type_text",
    }

    def __init__(self, belief_state) -> None:
        self.belief_state = belief_state

    def evaluate(self, action_name: str, risk_score: float = 0.0) -> PolicyDecision:
        action = str(action_name or "").strip().lower()
        risk = float(risk_score or 0.0)

        if action in self.HARD_DENY:
            return PolicyDecision(
                verdict=PolicyVerdict.DENY,
                reason=f"'{action}' is not allowed.",
                rule_name="hard_deny_list",
            )

        if action in self.ALWAYS_CONFIRM:
            return PolicyDecision(
                verdict=PolicyVerdict.REQUIRE_CONFIRM,
                reason=f"'{action}' always needs user confirmation.",
                rule_name="always_confirm_list",
            )

        if risk >= 0.95:
            return PolicyDecision(
                verdict=PolicyVerdict.DENY,
                reason="Risk score exceeds automatic execution threshold.",
                rule_name="risk_threshold_deny",
            )

        if risk >= 0.6 or bool(self.belief_state.should_ask_user(action, risk)):
            return PolicyDecision(
                verdict=PolicyVerdict.REQUIRE_CONFIRM,
                reason="Risk score requires user confirmation.",
                rule_name="risk_threshold_confirm",
            )

        return PolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            reason="Action is within safe operating bounds.",
            rule_name="default_allow",
        )

    def check(self, context, action_name: str, params=None) -> PolicyDecision:
        del params
        risk_score = float(getattr(context, "risk_score", 0.0) or 0.0)
        return self.evaluate(action_name, risk_score=risk_score)


__all__ = ["AutonomyPolicy", "PolicyDecision", "PolicyVerdict"]
