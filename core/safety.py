"""
core/safety.py
──────────────
Command Safety Gate (Session 6).
Enforces allowlists/denylists and dry-run policies for commands.
"""

import logging

logger = logging.getLogger(__name__)

class CommandSafetyGate:
    """
    The firewall for Jarvis Actions.
    Determines if a command is safe to execute.
    """
    
    # SAFE: Read-only or benign session commands
    ALLOWLIST = {
        "help", "status", "whoami", "exit", "quit", "bye", 
        "clear", "version", "uptime"
    }

    # DANGEROUS: File system or system-level modifications (for future proofing)
    DENYLIST = {
        "exec", "rm", "delete", "format", "sudo", 
        "install", "update", "upload", "download"
    }

    def __init__(self):
        pass

    def verify_command(self, command_str: str) -> dict:
        """
        Analyzes a command string.
        Returns: { "allowed": bool, "reason": str, "dry_run": str }
        """
        cmd_root = command_str.lower().split()[0]

        # 1. Check Denylist (Fail Fast)
        if cmd_root in self.DENYLIST:
            return {
                "allowed": False,
                "reason": f"Command '{cmd_root}' is explicitly BANNED.",
                "dry_run": None
            }

        # 2. Check Allowlist
        if cmd_root in self.ALLOWLIST:
            return {
                "allowed": True,
                "reason": "Command is in the safe allowlist.",
                "dry_run": f"System will execute internal routine: {cmd_root}"
            }

        # 3. Unknown Commands (Default Deny)
        return {
            "allowed": False,
            "reason": f"Command '{cmd_root}' is not recognized as safe.",
            "dry_run": None
        }
