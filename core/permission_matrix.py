"""Compatibility permission matrix built on top of the risk config."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PermissionResult:
    blocked_actions: list[str] = field(default_factory=list)
    confirmation_actions: list[str] = field(default_factory=list)

    @property
    def has_blocked(self) -> bool:
        return bool(self.blocked_actions)

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.confirmation_actions)


class PermissionMatrix:
    def __init__(self, config=None) -> None:
        self.config = config

    def evaluate(self, actions: list[str]) -> PermissionResult:
        blocked = self._parse_csv("risk", "blocked_actions")
        if not blocked:
            blocked = self._parse_csv("risk", "critical_actions")
        if not blocked:
            blocked = self._parse_csv("risk", "forbidden_actions")

        confirmation = self._parse_csv("risk", "user_confirmed_actions")
        if not confirmation:
            confirmation = self._parse_csv("risk", "high_risk_actions")

        normalized = [str(action).strip().lower() for action in actions if str(action).strip()]
        blocked_actions = [action for action in normalized if action in blocked]
        confirmation_actions = [
            action for action in normalized if action in confirmation and action not in blocked_actions
        ]
        return PermissionResult(
            blocked_actions=blocked_actions,
            confirmation_actions=confirmation_actions,
        )

    def _parse_csv(self, section: str, key: str) -> set[str]:
        if self.config is None:
            return set()
        try:
            raw = self.config.get(section, key, fallback="")
        except Exception:
            return set()
        return {item.strip().lower() for item in raw.split(",") if item.strip()}


__all__ = ["PermissionMatrix", "PermissionResult"]
