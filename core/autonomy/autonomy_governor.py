"""
AutonomyGovernor — enforces permission levels for tool execution dynamically.
Conforms to Rule 3.1 by avoiding hardcoded lists of tool names.
"""

import logging
import threading
from enum import IntEnum
from typing import Any

logger = logging.getLogger("Jarvis.AutonomyGovernor")


class AutonomyLevel(IntEnum):
    CHAT_ONLY = 0
    SUGGEST_ONLY = 1
    READ_ONLY = 2
    WRITE_WITH_CONFIRM = 3
    AUTONOMOUS = 4


class AutonomyGovernor:
    def __init__(self, level: int = 1, registry: Any = None):
        self.level = AutonomyLevel(level)
        self.registry = registry
        self.read_only_tools: set[str] = set()
        self.write_tools: set[str] = set()
        self._cache_is_write: dict[str, bool] = {}
        self._cache_is_known: dict[str, bool] = {}
        self._lock = threading.Lock()
        logger.info(f"Autonomy level set to: LEVEL_{self.level} ({self.level.name})")

    def register_read_only_tool(self, tool_name: str) -> None:
        """Dynamically register a tool as read-only."""
        name_clean = tool_name.strip().lower()
        with self._lock:
            self.read_only_tools.add(name_clean)
            self.write_tools.discard(name_clean)
            self._cache_is_write.pop(name_clean, None)
            self._cache_is_known.pop(name_clean, None)
        logger.debug("Dynamically registered read-only tool: %s", tool_name)

    def register_write_tool(self, tool_name: str) -> None:
        """Dynamically register a tool as a write tool."""
        name_clean = tool_name.strip().lower()
        with self._lock:
            self.write_tools.add(name_clean)
            self.read_only_tools.discard(name_clean)
            self._cache_is_write.pop(name_clean, None)
            self._cache_is_known.pop(name_clean, None)
        logger.debug("Dynamically registered write tool: %s", tool_name)

    def _is_known_tool(self, tool_name: str) -> bool:
        name_clean = tool_name.strip().lower()
        is_known = self._cache_is_known.get(name_clean)
        if is_known is not None:
            return is_known
            
        with self._lock:
            is_known = self._cache_is_known.get(name_clean)
            if is_known is not None:
                return is_known
                
            is_known = False
            if name_clean in self.read_only_tools or name_clean in self.write_tools:
                is_known = True
            elif self.registry and self.registry.get(name_clean) is not None:
                is_known = True
                
            if len(self._cache_is_known) > 1000:
                self._cache_is_known.clear()
            self._cache_is_known[name_clean] = is_known
            return is_known

    def _is_write_tool(self, tool_name: str) -> bool:
        name_clean = tool_name.strip().lower()
        is_write = self._cache_is_write.get(name_clean)
        if is_write is not None:
            return is_write
            
        with self._lock:
            is_write = self._cache_is_write.get(name_clean)
            if is_write is not None:
                return is_write
                
            if name_clean in self.write_tools:
                if len(self._cache_is_write) > 1000:
                    self._cache_is_write.clear()
                self._cache_is_write[name_clean] = True
                return True
            if name_clean in self.read_only_tools:
                if len(self._cache_is_write) > 1000:
                    self._cache_is_write.clear()
                self._cache_is_write[name_clean] = False
                return False

            if self.registry:
                cap = self.registry.get(name_clean)
                if cap:
                    if hasattr(cap, "is_write_operation"):
                        val = cap.is_write_operation
                        res = val() if callable(val) else bool(val)
                        if len(self._cache_is_write) > 1000:
                            self._cache_is_write.clear()
                        self._cache_is_write[name_clean] = res
                        return res
                    if hasattr(cap, "is_write"):
                        res = bool(cap.is_write)
                        if len(self._cache_is_write) > 1000:
                            self._cache_is_write.clear()
                        self._cache_is_write[name_clean] = res
                        return res

            # Fallback safe keyword-based check to avoid hardcoding tool name strings
            write_keywords = {
                "write", "delete", "remove", "unlink", "launch", "execute", 
                "run", "click", "type", "press", "move", "drag", "scroll", 
                "send", "add", "create", "mark", "clear", "play", "toggle", 
                "turn_on", "turn_off", "set_thermostat", "call_service", 
                "double_click", "right_click", "focus_window", "clipboard_set",
                "clipboard_paste", "hotkey"
            }
            res = any(kw in name_clean for kw in write_keywords)
            if len(self._cache_is_write) > 1000:
                self._cache_is_write.clear()
            self._cache_is_write[name_clean] = res
            return res

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

        if self.level >= AutonomyLevel.AUTONOMOUS:
            return True, f"Write tool '{tool_name}' approved at LEVEL_4 (fully autonomous)."

        if self.level >= AutonomyLevel.WRITE_WITH_CONFIRM:
            return True, f"Write tool '{tool_name}' approved at LEVEL_3 (confirmation required separately)."
        
        return False, f"Write tool '{tool_name}' blocked at LEVEL_{self.level} (need LEVEL_3)."

    def requires_confirmation(self, tool_name: str) -> bool:
        """Write tools at LEVEL_3 always need explicit user confirmation."""
        if self.level >= AutonomyLevel.AUTONOMOUS:
            return False
        return self.level == AutonomyLevel.WRITE_WITH_CONFIRM and self._is_write_tool(tool_name)

    def escalate(self, new_level: int) -> bool:
        """Temporarily escalate autonomy (user must consent upstream)."""
        if new_level > AutonomyLevel.AUTONOMOUS:
            logger.warning("Escalation above LEVEL_4 is not permitted.")
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
            4: "Fully autonomous — can run any allowed tool without confirmation.",
        }
        return f"LEVEL_{self.level}: {descriptions.get(self.level, 'Unknown')}"
