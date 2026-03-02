"""
JARVIS Risk Evaluator - Session 5
Trusted Core: No LLM can execute code without passing this check.
Table-based scoring — deterministic and auditable.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("JARVIS.RiskEvaluator")


@dataclass
class RiskResult:
    score: int            # 0-100
    level: str            # LOW / MEDIUM / HIGH / CRITICAL
    approved: bool
    reason: str
    flags: list[str]


# Risk scoring table: (pattern, score_addition, flag_label)
RISK_TABLE = [
    # File System
    (r'\brm\s+-rf\b',          90, "DESTRUCTIVE_DELETE"),
    (r'\bformat\b',            85, "FORMAT_DRIVE"),
    (r'\bshutil\.rmtree\b',    75, "RECURSIVE_DELETE"),
    (r'\bos\.remove\b',        30, "FILE_DELETE"),
    (r'\bopen\(.+[\'"]w[\'"]\)', 20, "FILE_WRITE"),

    # Network
    (r'\brequests\.',          15, "NETWORK_CALL"),
    (r'\bsmtplib\b',           40, "EMAIL_SEND"),
    (r'\bsocket\b',            25, "RAW_SOCKET"),

    # Process / System
    (r'\bsubprocess\b',        50, "SUBPROCESS"),
    (r'\bos\.system\b',        55, "OS_SYSTEM"),
    (r'\bexec\(',              70, "EXEC_CALL"),
    (r'\beval\(',              70, "EVAL_CALL"),
    (r'\b__import__\b',        60, "DYNAMIC_IMPORT"),

    # Sensitive Data
    (r'\bpassword\b',          20, "PASSWORD_REFERENCE"),
    (r'\bapi.?key\b',          25, "API_KEY_REFERENCE"),

    # Registry / Admin
    (r'\bwinreg\b',            65, "REGISTRY_ACCESS"),
    (r'\bctypes\b',            50, "CTYPES_USAGE"),
]

THRESHOLDS = {
    "LOW":      (0,  25),
    "MEDIUM":   (26, 50),
    "HIGH":     (51, 74),
    "CRITICAL": (75, 100),
}

AUTO_APPROVE_MAX = 25   # Score <= this is auto-approved
REQUIRE_CONFIRM_MAX = 50  # Score <= this requires user confirmation
# Score > 50 is BLOCKED


class RiskEvaluator:
    def __init__(self, auto_approve_threshold: int = AUTO_APPROVE_MAX):
        self.auto_approve_threshold = auto_approve_threshold
        logger.info(f"RiskEvaluator ready. Auto-approve threshold: {auto_approve_threshold}")

    def evaluate(self, code_or_plan: str, context: str = "") -> RiskResult:
        """Evaluate risk of a plan or code snippet."""
        flags = []
        total_score = 0

        text = (code_or_plan + " " + context).lower()

        for pattern, score, flag in RISK_TABLE:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(flag)
                total_score += score
                logger.debug(f"Risk flag triggered: {flag} (+{score})")

        # Cap at 100
        total_score = min(total_score, 100)

        # Determine level
        level = "LOW"
        for lvl, (low, high) in THRESHOLDS.items():
            if low <= total_score <= high:
                level = lvl
                break

        # Determine approval
        approved = total_score <= self.auto_approve_threshold
        reason = self._build_reason(total_score, level, flags)

        result = RiskResult(
            score=total_score,
            level=level,
            approved=approved,
            reason=reason,
            flags=flags
        )

        logger.info(f"Risk evaluation: Score={total_score} Level={level} Approved={approved} Flags={flags}")
        return result

    def _build_reason(self, score: int, level: str, flags: list[str]) -> str:
        if not flags:
            return "No risk patterns detected. Safe to execute."
        flag_str = ", ".join(flags)
        if score <= AUTO_APPROVE_MAX:
            return f"Low-risk flags detected ({flag_str}). Auto-approved."
        elif score <= REQUIRE_CONFIRM_MAX:
            return f"Medium risk detected ({flag_str}). User confirmation required."
        else:
            return f"HIGH/CRITICAL risk detected ({flag_str}). Execution BLOCKED."

    def require_confirmation(self, result: RiskResult) -> bool:
        """Returns True if user must confirm before execution."""
        return AUTO_APPROVE_MAX < result.score <= REQUIRE_CONFIRM_MAX

    def is_blocked(self, result: RiskResult) -> bool:
        return result.score > REQUIRE_CONFIRM_MAX
