"""
core/risk_evaluator.py — Stateless, table-driven risk scorer.

Given a list of action strings from the planner, returns a RiskResult.
No side effects. No LLM calls. Deterministic.

Risk levels (ascending): LOW < MEDIUM < HIGH < FORBIDDEN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence


class RiskLevel(IntEnum):
    LOW      = 0
    MEDIUM   = 1
    HIGH     = 2
    FORBIDDEN = 3

    def label(self) -> str:
        return self.name


@dataclass(frozen=True)
class RiskResult:
    level: RiskLevel
    blocking_actions: list[str]      = field(default_factory=list)
    high_risk_actions: list[str]     = field(default_factory=list)
    reasons: list[str]               = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.level == RiskLevel.FORBIDDEN

    @property
    def requires_confirmation(self) -> bool:
        return self.level >= RiskLevel.MEDIUM

    def summary(self) -> str:
        parts = [f"Risk: {self.level.label()}"]
        if self.blocking_actions:
            parts.append(f"BLOCKED: {', '.join(self.blocking_actions)}")
        if self.high_risk_actions:
            parts.append(f"HIGH: {', '.join(self.high_risk_actions)}")
        if self.reasons:
            parts.append(" | ".join(self.reasons))
        return " — ".join(parts)


class RiskEvaluator:
    """
    Evaluate risk of a plan.

    Action classification is loaded from config but falls back to sensible
    built-in defaults so the evaluator always works even without config.
    """

    _DEFAULT_FORBIDDEN = frozenset({
        "shell_exec", "shell", "exec", "subprocess",
        "file_delete", "delete_file", "rm", "rmdir",
        "registry_write", "registry_delete",
        "network_request", "http_get", "http_post", "curl", "wget",
        "format_disk", "wipe",
    })

    _DEFAULT_HIGH = frozenset({
        "file_write", "write_file", "save_file",
        "process_spawn", "spawn_process", "popen",
        "app_open",
        "serial_send", "serial_write",
        "physical_actuate", "actuate", "motor_command",
        "gui_click", "gui_type", "gui_hotkey",
        "system_config", "set_config", "env_write",
        "install_package", "pip_install",
    })

    _DEFAULT_MEDIUM = frozenset({
        "file_read", "read_file", "open_file",
        "screen_capture", "screenshot",
        "web_search",
        "ui_interaction", "click", "type_text", "key_press",
        "notification", "send_notification",
    })

    _DEFAULT_LOW = frozenset({
        "memory_read", "memory_write", "recall", "store_fact",
        "speak", "tts", "display", "print",
        "status", "health_check", "system_stats", "vision_analyze",
    })

    def __init__(self, config=None) -> None:
        self._forbidden = set(self._DEFAULT_FORBIDDEN)
        self._high      = set(self._DEFAULT_HIGH)
        self._medium    = set(self._DEFAULT_MEDIUM)
        self._low       = set(self._DEFAULT_LOW)

        if config is not None:
            self._load_config(config)

    def _load_config(self, config) -> None:
        def _parse(section: str, key: str) -> set[str]:
            raw = config.get(section, key, fallback="")
            return {a.strip().lower() for a in raw.split(",") if a.strip()}

        forbidden = _parse("risk", "forbidden_actions")
        high      = _parse("risk", "high_risk_actions")
        medium    = _parse("risk", "medium_risk_actions")
        low       = _parse("risk", "low_risk_actions")

        if forbidden: self._forbidden = forbidden
        if high:      self._high      = high
        if medium:    self._medium    = medium
        if low:       self._low       = low

    def evaluate(self, actions: Sequence[str]) -> RiskResult:
        """Evaluate a list of action names. Returns a RiskResult."""
        if not actions:
            return RiskResult(
                level=RiskLevel.LOW,
                reasons=["No actions — trivial plan"],
            )

        blocking:  list[str] = []
        high_risk: list[str] = []
        reasons:   list[str] = []
        max_level = RiskLevel.LOW

        for raw_action in actions:
            action = raw_action.strip().lower()

            if action in self._forbidden:
                blocking.append(action)
                max_level = RiskLevel.FORBIDDEN
                reasons.append(f"'{action}' is forbidden (L2/L3 blocked)")

            elif action in self._high:
                high_risk.append(action)
                if max_level < RiskLevel.HIGH:
                    max_level = RiskLevel.HIGH
                reasons.append(f"'{action}' is high-risk")

            elif action in self._medium:
                if max_level < RiskLevel.MEDIUM:
                    max_level = RiskLevel.MEDIUM
                reasons.append(f"'{action}' requires confirmation")

            elif action in self._low:
                pass  # stays LOW
            else:
                # Unknown action — treat as HIGH by default (conservative)
                high_risk.append(action)
                if max_level < RiskLevel.HIGH:
                    max_level = RiskLevel.HIGH
                reasons.append(f"'{action}' is unknown — treated as high-risk")

        return RiskResult(
            level=max_level,
            blocking_actions=blocking,
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
