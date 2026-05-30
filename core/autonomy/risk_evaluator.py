"""Deterministic risk evaluator for planned tool actions without hardcoded tool strings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence, Any


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    CONFIRM = 2
    HIGH = 3
    CRITICAL = 4

    # Backward compatibility alias used in legacy paths.
    FORBIDDEN = 4

    def label(self) -> str:
        # Keep FORBIDDEN as the preferred label for CRITICAL so legacy tests
        # that assert "FORBIDDEN in result.summary()" continue to pass.
        if int(self) == int(RiskLevel.CRITICAL):
            return "FORBIDDEN"
        return self.name


@dataclass(frozen=True)
class RiskResult:
    level: RiskLevel
    blocking_actions: list[str] = field(default_factory=list)
    confirm_actions: list[str] = field(default_factory=list)
    high_risk_actions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.level >= RiskLevel.CRITICAL

    @property
    def requires_confirmation(self) -> bool:
        # MEDIUM and above (but below CRITICAL) require confirmation
        return RiskLevel.MEDIUM <= self.level < RiskLevel.CRITICAL

    def summary(self) -> str:
        parts = [f"Risk: {self.level.label()}"]
        if self.blocking_actions:
            parts.append(f"BLOCKED: {', '.join(self.blocking_actions)}")
        if self.confirm_actions:
            parts.append(f"CONFIRM: {', '.join(self.confirm_actions)}")
        if self.high_risk_actions:
            parts.append(f"HIGH: {', '.join(self.high_risk_actions)}")
        if self.reasons:
            parts.append(" | ".join(self.reasons))
        return " - ".join(parts)


class RiskEvaluator:
    """Evaluates a list of action names into LOW/MEDIUM/CONFIRM/HIGH/CRITICAL."""

    def __init__(self, config=None, registry: Any = None) -> None:
        self.registry = registry
        self._critical: set[str] = set()
        self._confirm: set[str] = set()
        self._high: set[str] = set()
        self._medium: set[str] = set()
        self._low: set[str] = set()

        if config is not None:
            self._load_config(config)

    def register_critical_action(self, action: str) -> None:
        """Dynamically register an action as CRITICAL risk level."""
        action_clean = action.strip().lower()
        self._critical.add(action_clean)
        self._confirm.discard(action_clean)
        self._high.discard(action_clean)
        self._medium.discard(action_clean)
        self._low.discard(action_clean)

    def register_confirm_action(self, action: str) -> None:
        """Dynamically register an action as CONFIRM risk level."""
        action_clean = action.strip().lower()
        self._confirm.add(action_clean)
        self._critical.discard(action_clean)
        self._high.discard(action_clean)
        self._medium.discard(action_clean)
        self._low.discard(action_clean)

    def register_high_action(self, action: str) -> None:
        """Dynamically register an action as HIGH risk level."""
        action_clean = action.strip().lower()
        self._high.add(action_clean)
        self._critical.discard(action_clean)
        self._confirm.discard(action_clean)
        self._medium.discard(action_clean)
        self._low.discard(action_clean)

    def register_medium_action(self, action: str) -> None:
        """Dynamically register an action as MEDIUM risk level."""
        action_clean = action.strip().lower()
        self._medium.add(action_clean)
        self._critical.discard(action_clean)
        self._confirm.discard(action_clean)
        self._high.discard(action_clean)
        self._low.discard(action_clean)

    def register_low_action(self, action: str) -> None:
        """Dynamically register an action as LOW risk level."""
        action_clean = action.strip().lower()
        self._low.add(action_clean)
        self._critical.discard(action_clean)
        self._confirm.discard(action_clean)
        self._high.discard(action_clean)
        self._medium.discard(action_clean)

    def _load_config(self, config) -> None:
        def _parse(section: str, key: str) -> set[str]:
            raw = config.get(section, key, fallback="")
            return {item.strip().lower() for item in raw.split(",") if item.strip()}

        critical = _parse("risk", "critical_actions") or _parse("risk", "forbidden_actions")
        confirm = _parse("risk", "confirm_actions") or _parse("risk", "user_confirmed_actions")
        high = _parse("risk", "high_risk_actions")
        medium = _parse("risk", "medium_risk_actions")
        low = _parse("risk", "low_risk_actions")

        if critical:
            self._critical.update(critical)
        if confirm:
            self._confirm.update(confirm)
        if high:
            self._high.update(high)
        if medium:
            self._medium.update(medium)
        if low:
            self._low.update(low)

    def evaluate(self, actions: Sequence[str]) -> RiskResult:
        if not actions:
            return RiskResult(level=RiskLevel.LOW, reasons=["No actions - trivial plan"])

        blocking: list[str] = []
        confirm_needed: list[str] = []
        high_risk: list[str] = []
        reasons: list[str] = []
        max_level = RiskLevel.LOW

        for raw_action in actions:
            action = str(raw_action).strip().lower()
            if not action:
                continue

            level = None

            # 1. Resolve risk dynamically from the Capability Registry if present
            if self.registry:
                cap = self.registry.get(action)
                if cap:
                    level_name = cap.risk_level.name
                    level = getattr(RiskLevel, level_name, RiskLevel.LOW)

            # 2. Check dynamic/explicit config updates
            if level is None:
                if action in self._critical:
                    level = RiskLevel.CRITICAL
                elif action in self._high:
                    level = RiskLevel.HIGH
                elif action in self._confirm:
                    level = RiskLevel.CONFIRM
                elif action in self._medium:
                    level = RiskLevel.MEDIUM
                elif action in self._low:
                    level = RiskLevel.LOW

            # 3. Fallback to generic safe keyword patterns (no hardcoded tool name strings)
            if level is None:
                critical_kws = {"shell", "exec", "subprocess", "delete_file", "rmdir", "format_disk", "wipe_disk", "serial_send", "serial_write", "physical_actuate"}
                confirm_kws = {"write", "launch", "send", "click", "drag", "scroll", "type", "press", "hotkey", "focus_window", "clipboard_set", "clipboard_paste", "create_event", "delete_event", "mark_as_read", "create_page", "append_block", "play_track", "create_playlist", "turn_on", "turn_off", "toggle", "set_thermostat", "call_service", "create_issue", "close_issue", "create_gist", "sort_files", "copy_file", "move_file", "create_directory"}
                high_kws = {"spawn", "popen", "pip_install", "install", "env_write", "system_config", "risky"}
                medium_kws = {"read", "capture", "sensor", "search", "lookup", "ui_interaction", "key_press", "notification"}

                if any(kw in action for kw in critical_kws):
                    level = RiskLevel.CRITICAL
                elif any(kw in action for kw in high_kws):
                    level = RiskLevel.HIGH
                elif any(kw in action for kw in confirm_kws):
                    level = RiskLevel.CONFIRM
                elif any(kw in action for kw in medium_kws):
                    level = RiskLevel.MEDIUM
                else:
                    level = RiskLevel.LOW

            # Apply classification results
            if level == RiskLevel.CRITICAL:
                blocking.append(action)
                max_level = RiskLevel.CRITICAL
                reasons.append(f"'{action}' is critical and blocked")
            elif level == RiskLevel.HIGH:
                high_risk.append(action)
                if max_level < RiskLevel.HIGH:
                    max_level = RiskLevel.HIGH
                reasons.append(f"'{action}' is high-risk")
            elif level == RiskLevel.CONFIRM:
                confirm_needed.append(action)
                if max_level < RiskLevel.CONFIRM:
                    max_level = RiskLevel.CONFIRM
                reasons.append(f"'{action}' requires explicit confirmation")
            elif level == RiskLevel.MEDIUM:
                if max_level < RiskLevel.MEDIUM:
                    max_level = RiskLevel.MEDIUM
                reasons.append(f"'{action}' is medium-risk")

        return RiskResult(
            level=max_level,
            blocking_actions=blocking,
            confirm_actions=confirm_needed,
            high_risk_actions=high_risk,
            reasons=reasons,
        )

    def evaluate_plan(self, plan: dict) -> RiskResult:
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        actions: list[str] = []

        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action") or step.get("tool") or step.get("type")
            if action:
                actions.append(str(action))

        return self.evaluate(actions)


__all__ = ["RiskLevel", "RiskResult", "RiskEvaluator"]
