"""
Permission matrix for action-level policy enforcement.

This layer is separate from risk scoring:
- Risk answers "how dangerous is this?"
- Permission matrix answers "is this allowed and does it require confirmation?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class PermissionEvaluation:
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
        self._blocked_actions: set[str] = set()
        self._confirm_actions: set[str] = set()

        if config is not None:
            self._load_from_config(config)

        if not self._blocked_actions:
            self._blocked_actions = {
                "shell_exec",
                "file_delete",
                "format_disk",
                "wipe_disk",
                "registry_write",
            }
        if not self._confirm_actions:
            self._confirm_actions = {
                "file_write",
                "app_open",
                "serial_send",
                "physical_actuate",
                "gui_click",
                "gui_type",
                "gui_hotkey",
                "vision_click",
            }

    def _load_from_config(self, config) -> None:
        blocked = config.get("risk", "blocked_actions", fallback="").strip()
        confirmed = config.get("risk", "user_confirmed_actions", fallback="").strip()

        if blocked:
            self._blocked_actions = {
                action.strip().lower() for action in blocked.split(",") if action.strip()
            }

        if confirmed:
            self._confirm_actions = {
                action.strip().lower() for action in confirmed.split(",") if action.strip()
            }

        # Backward-compatibility: fall back to forbidden_actions when blocked_actions
        # is not explicitly defined.
        if not self._blocked_actions:
            forbidden = config.get("risk", "forbidden_actions", fallback="").strip()
            if forbidden:
                self._blocked_actions = {
                    action.strip().lower()
                    for action in forbidden.split(",")
                    if action.strip()
                }

    def evaluate(self, actions: Sequence[str]) -> PermissionEvaluation:
        blocked: list[str] = []
        confirm: list[str] = []
        for raw in actions:
            action = str(raw).strip().lower()
            if not action:
                continue
            if action in self._blocked_actions:
                blocked.append(action)
                continue
            if action in self._confirm_actions:
                confirm.append(action)

        return PermissionEvaluation(
            blocked_actions=blocked,
            confirmation_actions=confirm,
        )

    def is_blocked(self, action: str) -> bool:
        return action.strip().lower() in self._blocked_actions

    def requires_confirmation(self, action: str) -> bool:
        return action.strip().lower() in self._confirm_actions
