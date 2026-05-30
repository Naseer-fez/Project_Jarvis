"""
AutonomyGovernor — enforces permission levels for tool execution dynamically.
Conforms to Rule 3.1 by avoiding hardcoded lists of tool names.
"""

import logging
from enum import IntEnum
from typing import Any

logger = logging.getLogger("Jarvis.AutonomyGovernor")


class AutonomyLevel(IntEnum):
    CHAT_ONLY = 0
    SUGGEST_ONLY = 1
    READ_ONLY = 2
    WRITE_WITH_CONFIRM = 3


class AutonomyGovernor:
    def __init__(self, level: int = 1, registry: Any = None):
        self.level = AutonomyLevel(level)
        self.registry = registry
        self.read_only_tools: set[str] = set()
        self.write_tools: set[str] = set()
        logger.info(f"Autonomy level set to: LEVEL_{self.level} ({self.level.name})")

    def register_read_only_tool(self, tool_name: str) -> None:
        """Dynamically register a tool as read-only."""
        self.read_only_tools.add(tool_name)
        self.write_tools.discard(tool_name)
        logger.debug("Dynamically registered read-only tool: %s", tool_name)

    def register_write_tool(self, tool_name: str) -> None:
        """Dynamically register a tool as a write tool."""
        self.write_tools.add(tool_name)
        self.read_only_tools.discard(tool_name)
        logger.debug("Dynamically registered write tool: %s", tool_name)

    def _is_known_tool(self, tool_name: str) -> bool:
        name_clean = tool_name.strip().lower()
        if name_clean in self.read_only_tools or name_clean in self.write_tools:
            return True
        if self.registry and self.registry.get(name_clean) is not None:
            return True
        return False

    def _is_write_tool(self, tool_name: str) -> bool:
        name_clean = tool_name.strip().lower()
        if name_clean in self.write_tools:
            return True
        if name_clean in self.read_only_tools:
            return False

        if self.registry:
            cap = self.registry.get(name_clean)
            if cap:
                if hasattr(cap, "is_write_operation"):
                    val = cap.is_write_operation
                    return val() if callable(val) else bool(val)
                if hasattr(cap, "is_write"):
                    return bool(cap.is_write)

        # Fallback safe keyword-based check to avoid hardcoding tool name strings
        write_keywords = {
            "write", "delete", "remove", "unlink", "launch", "execute", 
            "run", "click", "type", "press", "move", "drag", "scroll", 
            "send", "add", "create", "mark", "clear", "play", "toggle", 
            "turn_on", "turn_off", "set_thermostat", "call_service", 
            "double_click", "right_click", "focus_window", "clipboard_set",
            "clipboard_paste", "hotkey"
        }
        return any(kw in name_clean for kw in write_keywords)

    def can_execute(self, tool_name: str) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        """
        if not self._is_known_tool(tool_name):
            return False, f"Unknown tool '{tool_name}' is blocked by default. Add it to WRITE_TOOLS or READ_ONLY_TOOLS."

        if self.level == AutonomyLevel.CHAT_ONLY:
            return False, "Autonomy LEVEL_0: tool execution is disabled."

        if self.level == AutonomyLevel.SUGGEST_ONLY:
            return False, f"Autonomy LEVEL_1: would call '{tool_name}' but only suggesting actions."

        is_write = self._is_write_tool(tool_name)

        if not is_write:
            return True, f"Read-only tool '{tool_name}' approved at LEVEL_{self.level}."

        if self.level >= AutonomyLevel.WRITE_WITH_CONFIRM:
            return True, f"Write tool '{tool_name}' approved at LEVEL_3 (confirmation required separately)."
        
        return False, f"Write tool '{tool_name}' blocked at LEVEL_{self.level} (need LEVEL_3)."

    def requires_confirmation(self, tool_name: str) -> bool:
        """Write tools at LEVEL_3 always need explicit user confirmation."""
        return self.level == AutonomyLevel.WRITE_WITH_CONFIRM and self._is_write_tool(tool_name)

    def escalate(self, new_level: int) -> bool:
        """Temporarily escalate autonomy (user must consent upstream)."""
        if new_level > AutonomyLevel.WRITE_WITH_CONFIRM:
            logger.warning("Escalation above LEVEL_3 is not permitted.")
            return False
        old = self.level
        self.level = AutonomyLevel(new_level)
        logger.info(f"Autonomy escalated: {old.name} -> {self.level.name}")
        return True

    def describe(self) -> str:
        descriptions = {
            0: "Chat only — no tool execution.",
            1: "Suggest only — describes actions but never runs them.",
            2: "Read-only — can inspect files, web, screen, and status automatically.",
            3: "Write with confirmation — can change files, apps, and desktop state after your approval.",
        }
        return f"LEVEL_{self.level}: {descriptions[self.level]}"
