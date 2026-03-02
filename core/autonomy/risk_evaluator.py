"""
Stateless, table-driven risk scorer.

Given a list of action strings from the planner, returns a RiskResult.
No side effects. No LLM calls. Deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    CONFIRM = 2
    HIGH = 3
    CRITICAL = 4
    # Backward-compatibility alias used by older tests/code paths.
    FORBIDDEN = 4

    def label(self) -> str:
        if int(self) == int(RiskLevel.CRITICAL):
            return "CRITICAL"
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
        return self.level >= RiskLevel.CONFIRM and not self.is_blocked

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
    """
    Evaluate risk of a plan.

    Action classification is loaded from config but falls back to sensible
    built-in defaults so the evaluator always works even without config.
    """

    _DEFAULT_CRITICAL = frozenset(
        {
            "shell_exec",
            "shell",
            "exec",
            "subprocess",
            "file_delete",
            "delete_file",
            "rm",
            "rmdir",
            "registry_write",
            "registry_delete",
            "format_disk",
            "wipe",
            "wipe_disk",
        }
    )

    _DEFAULT_CONFIRM = frozenset(
        {
            "write_file",
            "execute_shell",
            "launch_application",
            "send_email",
            "send_whatsapp",
            "add_calendar_event",
        }
    )

    _DEFAULT_HIGH = frozenset(
        {
            "file_write",
            "save_file",
            "process_spawn",
            "spawn_process",
            "popen",
            "app_open",
            "serial_send",
            "serial_write",
            "physical_actuate",
            "actuate",
            "motor_command",
            "vision_click",
            "gui_click",
            "gui_type",
            "gui_hotkey",
            "system_config",
            "set_config",
            "env_write",
            "install_package",
            "pip_install",
        }
    )

    _DEFAULT_MEDIUM = frozenset(
        {
            "file_read",
            "open_file",
            "screen_capture",
            "screenshot",
            "screen_understand",
            "sensor_read",
            "web_search",
            "ui_interaction",
            "click",
            "type_text",
            "key_press",
            "notification",
            "send_notification",
        }
    )

    _DEFAULT_LOW = frozenset(
        {
            "memory_read",
            "memory_write",
            "recall",
            "store_fact",
            "speak",
            "tts",
            "display",
            "print",
            "status",
            "health_check",
            "system_stats",
            "vision_analyze",
            "list_directory",
            "read_file",
            "search_code",
            "run_linter",
        }
    )

    def __init__(self, config=None) -> None:
        self._critical = set(self._DEFAULT_CRITICAL)
        self._confirm = set(self._DEFAULT_CONFIRM)
        self._high = set(self._DEFAULT_HIGH)
        self._medium = set(self._DEFAULT_MEDIUM)
        self._low = set(self._DEFAULT_LOW)

        if config is not None:
            self._load_config(config)

    def _load_config(self, config) -> None:
        def _parse(section: str, key: str) -> set[str]:
            raw = config.get(section, key, fallback="")
            return {a.strip().lower() for a in raw.split(",") if a.strip()}

        critical = _parse("risk", "critical_actions")
        if not critical:
            critical = _parse("risk", "forbidden_actions")
        confirm = _parse("risk", "confirm_actions")
        if not confirm:
            confirm = _parse("risk", "user_confirmed_actions")
        high = _parse("risk", "high_risk_actions")
        medium = _parse("risk", "medium_risk_actions")
        low = _parse("risk", "low_risk_actions")

        if critical:
            self._critical = critical
        if confirm:
            self._confirm = confirm
        if high:
            self._high = high
        if medium:
            self._medium = medium
        if low:
            self._low = low

    def evaluate(self, actions: Sequence[str]) -> RiskResult:
        """Evaluate a list of action names. Returns a RiskResult."""
        if not actions:
            return RiskResult(
                level=RiskLevel.LOW,
                reasons=["No actions - trivial plan"],
            )

        blocking: list[str] = []
        confirm_needed: list[str] = []
        high_risk: list[str] = []
        reasons: list[str] = []
        max_level = RiskLevel.LOW

        for raw_action in actions:
            action = raw_action.strip().lower()

            if action in self._critical:
                blocking.append(action)
                max_level = RiskLevel.CRITICAL
                reasons.append(f"'{action}' is critical and blocked")

            elif action in self._high:
                high_risk.append(action)
                if max_level < RiskLevel.HIGH:
                    max_level = RiskLevel.HIGH
                reasons.append(f"'{action}' is high-risk")

            elif action in self._confirm:
                confirm_needed.append(action)
                if max_level < RiskLevel.CONFIRM:
                    max_level = RiskLevel.CONFIRM
                reasons.append(f"'{action}' requires explicit confirmation")

            elif action in self._medium:
                if max_level < RiskLevel.MEDIUM:
                    max_level = RiskLevel.MEDIUM
                reasons.append(f"'{action}' is medium-risk")

            elif action in self._low:
                pass
            else:
                # Unknown action - treat as HIGH by default (conservative)
                high_risk.append(action)
                if max_level < RiskLevel.HIGH:
                    max_level = RiskLevel.HIGH
                reasons.append(f"'{action}' is unknown - treated as high-risk")

        return RiskResult(
            level=max_level,
            blocking_actions=blocking,
            confirm_actions=confirm_needed,
            high_risk_actions=high_risk,
            reasons=reasons,
        )

    def evaluate_plan(self, plan: dict) -> RiskResult:
        """Convenience: evaluate a full plan dict as produced by the planner."""
        steps = plan.get("steps", [])
        actions = []
        for step in steps:
            if isinstance(step, dict):
                action = step.get("action", step.get("type", ""))
                if action:
                    actions.append(action)
        return self.evaluate(actions)
