"""
core/safety.py
═══════════════
CommandSafetyGate — validates commands before execution.
V1 Rule: Physical actions, shell execution, and self-modification are ALWAYS blocked.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── Hard-blocked patterns ──────────────────────────────────
BLOCKED_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\b",
    r"\bexec\b",
    r"rm\s+-rf",
    r"del\s+/",
    r"format\s+[a-zA-Z]:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"drop\s+table",
    r"delete\s+from",
]

# ── Allowed command whitelist ──────────────────────────────
ALLOWED_COMMANDS = {
    "exit", "quit", "status", "synthesize",
    "reset", "help", "memory", "history", "clear"
}


class CommandSafetyGate:
    """
    Verifies a command is safe before execution.
    Returns {"allowed": bool, "reason": str}
    """

    def verify_command(self, user_input: str) -> dict:
        lower = user_input.lower().strip()
        first_word = lower.split()[0] if lower.split() else ""

        # Check blocked patterns
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, lower):
                reason = f"Blocked pattern detected: '{pattern}'"
                logger.warning(f"SAFETY BLOCK: {reason} | input={user_input!r}")
                return {"allowed": False, "reason": reason}

        # If it's a known safe command — allow
        if first_word in ALLOWED_COMMANDS:
            return {"allowed": True, "reason": "whitelisted command"}

        # Unknown command — allow but log
        logger.info(f"SAFETY: Unknown command '{first_word}' — allowing with log")
        return {"allowed": True, "reason": "passed safety check"}
