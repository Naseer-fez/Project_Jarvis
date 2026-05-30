"""Contracts for desktop actions, observations, and verification results."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class DesktopActionType(str, Enum):
    LAUNCH_APP = "launch_application"
    FOCUS_WINDOW = "focus_window"
    MOVE_MOUSE = "move_mouse"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    CLICK_TEXT_ON_SCREEN = "click_text_on_screen"
    CLICK_SCREEN_TARGET = "click_screen_target"
    DOUBLE_CLICK_SCREEN_TARGET = "double_click_screen_target"
    RIGHT_CLICK_SCREEN_TARGET = "right_click_screen_target"
    SCROLL = "scroll"
    DRAG = "drag"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOTKEY = "hotkey"
    CLIPBOARD_GET = "clipboard_get"
    CLIPBOARD_SET = "clipboard_set"
    CLIPBOARD_PASTE = "clipboard_paste"


class DesktopRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    CONFIRM = "confirm"
    HIGH = "high"
    BLOCKED = "blocked"


class DesktopActionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    NEEDS_APPROVAL = "needs_approval"


@dataclass(frozen=True)
class DesktopAction:
    action_type: DesktopActionType | str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expected_change: str = ""
    risk_tier: DesktopRiskTier | str | None = None
    requires_approval: bool | None = None
    action_id: str = field(default_factory=lambda: _new_id("act"))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_name(self) -> str:
        if isinstance(self.action_type, DesktopActionType):
            return self.action_type.value
        return str(self.action_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_name,
            "description": self.description,
            "params": dict(self.params),
            "expected_change": self.expected_change,
            "risk_tier": str(self.risk_tier.value if isinstance(self.risk_tier, DesktopRiskTier) else self.risk_tier or ""),
            "requires_approval": self.requires_approval,
            "metadata": dict(self.metadata),
        }


@dataclass
class DesktopActionResult:
    action_id: str
    action_type: str
    success: bool
    status: DesktopActionStatus
    output: str = ""
    error: str = ""
    risk_tier: DesktopRiskTier = DesktopRiskTier.MEDIUM
    audit_hash: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "success": self.success,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "risk_tier": self.risk_tier.value,
            "audit_hash": self.audit_hash,
            "duration_seconds": round(self.duration_seconds, 3),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScreenTarget:
    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DesktopObservation:
    observation_id: str = field(default_factory=lambda: _new_id("obs"))
    screenshot_path: str = ""
    screenshot_fingerprint: str = ""
    active_window: dict[str, Any] = field(default_factory=dict)
    ocr_text: str = ""
    targets: list[ScreenTarget] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    captured_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "screenshot_path": self.screenshot_path,
            "screenshot_fingerprint": self.screenshot_fingerprint,
            "active_window": dict(self.active_window),
            "ocr_text": self.ocr_text,
            "targets": [target.to_dict() for target in self.targets],
            "confidence": round(self.confidence, 3),
            "low_confidence_reason": self.low_confidence_reason,
            "metadata": dict(self.metadata),
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class DesktopChange:
    changed: bool
    confidence: float
    summary: str
    before_observation_id: str = ""
    after_observation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "before_observation_id": self.before_observation_id,
            "after_observation_id": self.after_observation_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ApprovalDecision:
    required: bool
    approved: bool
    reason: str = ""
    mode: str = "automatic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "approved": self.approved,
            "reason": self.reason,
            "mode": self.mode,
        }


__all__ = [
    "ApprovalDecision",
    "DesktopAction",
    "DesktopActionResult",
    "DesktopActionStatus",
    "DesktopActionType",
    "DesktopChange",
    "DesktopObservation",
    "DesktopRiskTier",
    "ScreenTarget",
]
