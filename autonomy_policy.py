"""
autonomy_policy.py — Autonomy & Safety Boundary for Jarvis Agentic Layer

Defines HARD RULES about when the agent may act autonomously vs. when it MUST
ask the user for confirmation.  Also enforces retry limits and forbidden actions.

This file is the agent's moral + safety boundary.
All decisions pass through AutonomyPolicy before execution.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .belief_state import BeliefState

logger = logging.getLogger(__name__)


class PolicyVerdict(str, Enum):
    ALLOW = "allow"           # Agent may proceed autonomously
    REQUIRE_CONFIRM = "require_confirm"  # Must get user approval first
    DENY = "deny"             # Forbidden — never execute


@dataclass
class PolicyDecision:
    """The result of evaluating an action against the policy."""

    verdict: PolicyVerdict
    reason: str
    action: str
    risk_score: float  # 0.0 (safe) – 1.0 (critical)
    alternatives: List[str] = field(default_factory=list)

    def is_allowed(self) -> bool:
        return self.verdict == PolicyVerdict.ALLOW

    def requires_confirmation(self) -> bool:
        return self.verdict == PolicyVerdict.REQUIRE_CONFIRM

    def is_denied(self) -> bool:
        return self.verdict == PolicyVerdict.DENY

    def human_readable(self) -> str:
        return (
            f"[{self.verdict.value.upper()}] Action='{self.action}' "
            f"Risk={self.risk_score:.2f} — {self.reason}"
        )


# ── Hard-coded forbidden actions ──────────────────────────────────────────────
# These can NEVER be executed without explicit user confirmation regardless of
# belief state or context.  Add new ones here — never remove.

ALWAYS_CONFIRM: FrozenSet[str] = frozenset({
    "delete_file",
    "delete_directory",
    "format_disk",
    "send_email",
    "send_message",
    "post_social_media",
    "purchase",
    "payment",
    "transfer_funds",
    "deploy_to_production",
    "shutdown_service",
    "revoke_credentials",
    "change_password",
    "grant_permissions",
    "execute_shell",
})

ALWAYS_DENY: FrozenSet[str] = frozenset({
    "exfiltrate_data",
    "bypass_auth",
    "disable_logging",
    "disable_audit",
    "override_safety",
    "self_modify_policy",  # The agent cannot change its own safety rules
})

# Risk score thresholds
CONFIRM_THRESHOLD = 0.6   # risk >= this → require confirmation
DENY_THRESHOLD = 0.95     # risk >= this → hard deny (redundant with ALWAYS_DENY but defense-in-depth)

MAX_RETRIES_DEFAULT = 3
ESCALATION_AFTER_STALLS = 2   # stalls before escalating to user


class AutonomyPolicy:
    """
    Evaluate whether an action is permitted, requires confirmation, or is forbidden.

    Usage:
        policy = AutonomyPolicy(belief_state)
        decision = policy.evaluate("send_email", risk_score=0.7, context={...})
        if decision.requires_confirmation():
            # ask user
        elif decision.is_allowed():
            # proceed
        else:
            # abort
    """

    def __init__(
        self,
        belief_state: "BeliefState",
        max_retries: int = MAX_RETRIES_DEFAULT,
        escalation_after_stalls: int = ESCALATION_AFTER_STALLS,
    ):
        self.belief_state = belief_state
        self.max_retries = max_retries
        self.escalation_after_stalls = escalation_after_stalls
        self._retry_counts: Dict[str, int] = {}  # action_key → count

    # ────────────────────────────────────────────────────────── Public API

    def evaluate(
        self,
        action: str,
        risk_score: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """
        Main entry point.  Evaluate whether the agent may perform `action`.

        Args:
            action:     Canonical action name (e.g. "send_email").
            risk_score: 0.0–1.0, estimated by the risk evaluator.
            context:    Optional metadata for logging/reasoning.

        Returns:
            PolicyDecision with verdict and human-readable reason.
        """
        context = context or {}
        action_lower = action.lower().strip()

        # ── 1. Hard deny ───────────────────────────────────────────────
        if action_lower in ALWAYS_DENY:
            return PolicyDecision(
                verdict=PolicyVerdict.DENY,
                reason=f"Action '{action}' is unconditionally forbidden by policy.",
                action=action,
                risk_score=1.0,
            )

        # ── 2. Risk score hard deny ────────────────────────────────────
        if risk_score >= DENY_THRESHOLD:
            return PolicyDecision(
                verdict=PolicyVerdict.DENY,
                reason=f"Risk score {risk_score:.2f} exceeds hard deny threshold {DENY_THRESHOLD}.",
                action=action,
                risk_score=risk_score,
            )

        # ── 3. Always-confirm list ────────────────────────────────────
        if action_lower in ALWAYS_CONFIRM:
            return PolicyDecision(
                verdict=PolicyVerdict.REQUIRE_CONFIRM,
                reason=f"Action '{action}' always requires user confirmation.",
                action=action,
                risk_score=risk_score,
            )

        # ── 4. Risk-based confirm ─────────────────────────────────────
        if risk_score >= CONFIRM_THRESHOLD:
            return PolicyDecision(
                verdict=PolicyVerdict.REQUIRE_CONFIRM,
                reason=(
                    f"Risk score {risk_score:.2f} ≥ threshold {CONFIRM_THRESHOLD}. "
                    "Confirmation required."
                ),
                action=action,
                risk_score=risk_score,
            )

        # ── 5. Belief-based check ─────────────────────────────────────
        if self.belief_state.should_ask_user():
            return PolicyDecision(
                verdict=PolicyVerdict.REQUIRE_CONFIRM,
                reason=(
                    f"Belief state signals low confidence or low risk tolerance "
                    f"(confidence={self.belief_state.get('agent_confidence'):.2f}, "
                    f"risk_tolerance={self.belief_state.get('risk_tolerance'):.2f})."
                ),
                action=action,
                risk_score=risk_score,
            )

        # ── 6. Allowed ────────────────────────────────────────────────
        logger.debug("Policy ALLOW: '%s' (risk=%.2f)", action, risk_score)
        return PolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            reason="Action passes all policy checks.",
            action=action,
            risk_score=risk_score,
        )

    def check_retry(self, action_key: str) -> bool:
        """
        Returns True if the agent is allowed to retry this action again.
        Increments internal counter.  Returns False once max_retries is hit.
        """
        count = self._retry_counts.get(action_key, 0)
        if count >= self.max_retries:
            logger.warning(
                "Retry limit reached for '%s' (%d/%d). Stopping.",
                action_key,
                count,
                self.max_retries,
            )
            return False
        self._retry_counts[action_key] = count + 1
        logger.debug("Retry %d/%d for '%s'.", count + 1, self.max_retries, action_key)
        return True

    def reset_retry(self, action_key: str) -> None:
        """Reset retry counter on success."""
        self._retry_counts.pop(action_key, None)

    def should_escalate(self, stall_count: int) -> bool:
        """True if the agent has stalled enough times to escalate to the user."""
        return stall_count >= self.escalation_after_stalls

    def explain(self) -> str:
        """Return a human-readable summary of the current policy configuration."""
        lines = [
            "── Autonomy Policy ─────────────────────────────",
            f"Max retries            : {self.max_retries}",
            f"Escalate after stalls  : {self.escalation_after_stalls}",
            f"Confirm threshold      : risk ≥ {CONFIRM_THRESHOLD}",
            f"Deny threshold         : risk ≥ {DENY_THRESHOLD}",
            f"Always-confirm actions : {len(ALWAYS_CONFIRM)}",
            f"Always-deny actions    : {len(ALWAYS_DENY)}",
            "Retry counts (active)  :",
        ]
        for k, v in self._retry_counts.items():
            lines.append(f"  {k}: {v}/{self.max_retries}")
        return "\n".join(lines)
