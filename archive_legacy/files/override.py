"""
core/autonomy/override.py

Human Override Protocol — the authoritative mechanism for humans to
intervene in agent execution.

Capabilities:
- Emergency stop (immediate, persistent)
- Approve or deny a pending action
- "Never do this again" permanent bans
- One-shot permissions (allow once, then revert)
- Full override log for audit

Design:
  All override records are append-only.  Nothing is ever deleted.
  The agent checks this module BEFORE autonomy_policy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OverrideType(str, Enum):
    EMERGENCY_STOP    = "emergency_stop"     # halt everything immediately
    APPROVE           = "approve"            # allow a specific pending action
    DENY              = "deny"               # block a specific pending action
    PERMANENT_BAN     = "permanent_ban"      # never allow this action again
    ONE_SHOT_PERMIT   = "one_shot_permit"    # allow once, then requires approval again
    RESUME            = "resume"             # lift an emergency stop


@dataclass
class OverrideRecord:
    """Immutable record of one human override instruction."""

    override_id: str
    override_type: OverrideType
    issued_by: str                          # human identifier / username
    target: str                             # action name, goal_id, or "*" for global
    reason: str
    expires_at: Optional[datetime]          # None = never expires
    consumed: bool = False                  # True once a one-shot permit is used

    issued_at: datetime = field(default_factory=_utcnow)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return _utcnow() > self.expires_at

    @property
    def is_active(self) -> bool:
        return not self.is_expired and not self.consumed

    def consume(self) -> None:
        """Mark a one-shot permit as used."""
        self.consumed = True

    def to_dict(self) -> dict:
        return {
            "override_id": self.override_id,
            "override_type": self.override_type.value,
            "issued_by": self.issued_by,
            "target": self.target,
            "reason": self.reason,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "consumed": self.consumed,
            "issued_at": self.issued_at.isoformat(),
            "is_active": self.is_active,
        }


class HumanOverrideProtocol:
    """
    Central registry of all human override instructions.

    The agent should consult this *before* the autonomy policy.

    Usage:
        hop = HumanOverrideProtocol()

        # Emergency stop
        hop.emergency_stop(issued_by="operator", reason="Runaway loop detected")
        hop.is_stopped()  # True

        # Lift stop
        hop.resume(issued_by="operator", reason="Issue resolved")
        hop.is_stopped()  # False

        # Permanent ban
        hop.permanent_ban("delete_all_files", issued_by="admin")

        # One-shot permit
        hop.one_shot_permit("send_mass_email", issued_by="alice")
        hop.check_one_shot("send_mass_email")  # True (and consumes permit)
        hop.check_one_shot("send_mass_email")  # False (permit consumed)
    """

    def __init__(self) -> None:
        self._records: list[OverrideRecord] = []

    # ── Emergency stop / resume ──────────────────────────────────────────

    def emergency_stop(self, issued_by: str, reason: str = "") -> OverrideRecord:
        return self._add(OverrideType.EMERGENCY_STOP, "*", issued_by, reason)

    def resume(self, issued_by: str, reason: str = "") -> OverrideRecord:
        return self._add(OverrideType.RESUME, "*", issued_by, reason)

    def is_stopped(self) -> bool:
        """
        Returns True if an un-lifted emergency stop is in effect.
        A RESUME record lifts the most recent stop.
        """
        for record in reversed(self._records):
            if record.override_type == OverrideType.EMERGENCY_STOP:
                return True
            if record.override_type == OverrideType.RESUME:
                return False
        return False

    # ── Approve / deny ───────────────────────────────────────────────────

    def approve(
        self,
        target: str,
        issued_by: str,
        reason: str = "",
        expires_at: Optional[datetime] = None,
    ) -> OverrideRecord:
        return self._add(OverrideType.APPROVE, target, issued_by, reason, expires_at)

    def deny(
        self,
        target: str,
        issued_by: str,
        reason: str = "",
        expires_at: Optional[datetime] = None,
    ) -> OverrideRecord:
        return self._add(OverrideType.DENY, target, issued_by, reason, expires_at)

    def check_approval(self, target: str) -> Optional[bool]:
        """
        Check if there is an active approval or denial for target.
        Returns True (approved), False (denied), or None (no override).
        Searches most-recent-first.
        """
        for record in reversed(self._records):
            if record.target not in (target, "*"):
                continue
            if not record.is_active:
                continue
            if record.override_type == OverrideType.APPROVE:
                return True
            if record.override_type == OverrideType.DENY:
                return False
        return None

    # ── Permanent ban ────────────────────────────────────────────────────

    def permanent_ban(self, target: str, issued_by: str, reason: str = "") -> OverrideRecord:
        return self._add(OverrideType.PERMANENT_BAN, target, issued_by, reason, expires_at=None)

    def is_permanently_banned(self, target: str) -> bool:
        return any(
            r.override_type == OverrideType.PERMANENT_BAN
            and r.target == target
            for r in self._records
        )

    # ── One-shot permits ─────────────────────────────────────────────────

    def one_shot_permit(
        self,
        target: str,
        issued_by: str,
        reason: str = "",
        expires_at: Optional[datetime] = None,
    ) -> OverrideRecord:
        return self._add(OverrideType.ONE_SHOT_PERMIT, target, issued_by, reason, expires_at)

    def check_one_shot(self, target: str) -> bool:
        """
        Return True and consume the permit if one exists.
        Returns False if no valid permit found.
        """
        for record in self._records:
            if (
                record.override_type == OverrideType.ONE_SHOT_PERMIT
                and record.target == target
                and record.is_active
            ):
                record.consume()
                return True
        return False

    # ── Audit log ────────────────────────────────────────────────────────

    def log(self) -> list[OverrideRecord]:
        return list(self._records)

    def log_for(self, target: str) -> list[OverrideRecord]:
        return [r for r in self._records if r.target in (target, "*")]

    def snapshot(self) -> list[dict]:
        return [r.to_dict() for r in self._records]

    # ── Internal ─────────────────────────────────────────────────────────

    def _add(
        self,
        override_type: OverrideType,
        target: str,
        issued_by: str,
        reason: str,
        expires_at: Optional[datetime] = None,
    ) -> OverrideRecord:
        record = OverrideRecord(
            override_id=str(uuid.uuid4()),
            override_type=override_type,
            issued_by=issued_by,
            target=target,
            reason=reason,
            expires_at=expires_at,
        )
        self._records.append(record)
        return record

