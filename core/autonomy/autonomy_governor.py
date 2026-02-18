"""
AutonomyGovernor — enforces permission levels for tool execution.

LEVEL_0: chat_only          → no tool calls, pure conversation
LEVEL_1: suggest_only       → describe what it would do, but never execute
LEVEL_2: read_only_tools    → execute read-only tools automatically
LEVEL_3: write_with_confirm → execute write tools after user confirmation
"""

import logging
from enum import IntEnum

logger = logging.getLogger("Jarvis.AutonomyGovernor")

READ_ONLY_TOOLS = {"get_time", "get_system_stats", "list_directory", "read_file", "search_memory"}
WRITE_TOOLS = {"write_file_safe", "log_event"}


class AutonomyLevel(IntEnum):
    CHAT_ONLY = 0
    SUGGEST_ONLY = 1
    READ_ONLY = 2
    WRITE_WITH_CONFIRM = 3


class AutonomyGovernor:
    def __init__(self, level: int = 1):
        self.level = AutonomyLevel(level)
        logger.info(f"Autonomy level set to: LEVEL_{self.level} ({self.level.name})")

    def can_execute(self, tool_name: str) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        """
        if self.level == AutonomyLevel.CHAT_ONLY:
            return False, "Autonomy LEVEL_0: tool execution is disabled."

        if self.level == AutonomyLevel.SUGGEST_ONLY:
            return False, f"Autonomy LEVEL_1: would call '{tool_name}' but only suggesting actions."

        if tool_name in READ_ONLY_TOOLS:
            return True, f"Read-only tool '{tool_name}' approved at LEVEL_{self.level}."

        if tool_name in WRITE_TOOLS:
            if self.level >= AutonomyLevel.WRITE_WITH_CONFIRM:
                return True, f"Write tool '{tool_name}' approved at LEVEL_3 (confirmation required separately)."
            else:
                return False, f"Write tool '{tool_name}' blocked at LEVEL_{self.level} (need LEVEL_3)."

        # Unknown tools — block by default
        return False, f"Unknown tool '{tool_name}' is blocked by default. Add it to WRITE_TOOLS or READ_ONLY_TOOLS."

    def requires_confirmation(self, tool_name: str) -> bool:
        """Write tools at LEVEL_3 always need explicit user confirmation."""
        return self.level == AutonomyLevel.WRITE_WITH_CONFIRM and tool_name in WRITE_TOOLS

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
            2: "Read-only — can query system/files/memory automatically.",
            3: "Write with confirmation — can modify files/memory after your approval.",
        }
        return f"LEVEL_{self.level}: {descriptions[self.level]}"

