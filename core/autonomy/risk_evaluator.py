"""
RiskEvaluator — heuristic weighted-sum risk scoring.

Risk inputs:
  - intent_risk:   inferred from the user's goal description
  - tool_risk:     based on the specific tool being used
  - profile_risk:  based on configured autonomy level
  - environment_risk: based on operational context

Risk thresholds:
  0.00 – 0.40 → SAFE (execute)
  0.40 – 0.60 → REVIEW (log warning)
  0.60 – 0.75 → CONFIRM (require user OK)
  0.75+       → BLOCK (refuse)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("Jarvis.RiskEvaluator")


class RiskLevel(Enum):
    SAFE = "safe"
    REVIEW = "review"
    CONFIRM = "confirm"
    BLOCK = "block"


# Tool-level base risk scores
TOOL_RISK_MAP: dict[str, float] = {
    # Read-only tools — very low risk
    "get_time": 0.0,
    "get_system_stats": 0.05,
    "list_directory": 0.05,
    "read_file": 0.1,
    "search_memory": 0.05,
    # Write tools — elevated risk
    "write_file_safe": 0.6,
    "log_event": 0.2,
    # Unknown tools
    "__unknown__": 0.8,
}

# Keywords that raise intent risk
DANGEROUS_INTENT_KEYWORDS = [
    "delete", "remove", "erase", "overwrite", "shutdown", "reboot",
    "format", "kill", "terminate", "sudo", "admin", "password", "secret",
]

ELEVATED_INTENT_KEYWORDS = [
    "write", "modify", "update", "change", "edit", "create", "install",
]

WEIGHTS = {
    "intent_risk": 0.30,
    "tool_risk": 0.40,
    "profile_risk": 0.20,
    "environment_risk": 0.10,
}

THRESHOLDS = {
    RiskLevel.SAFE: 0.40,
    RiskLevel.REVIEW: 0.60,
    RiskLevel.CONFIRM: 0.75,
}


@dataclass
class RiskReport:
    intent_risk: float
    tool_risk: float
    profile_risk: float
    environment_risk: float
    composite_score: float
    level: RiskLevel
    explanation: str


class RiskEvaluator:
    def __init__(self, autonomy_level: int = 1):
        self.autonomy_level = autonomy_level  # 0-3

    def evaluate(
        self,
        goal: str,
        tool_name: Optional[str] = None,
        step_description: str = "",
    ) -> RiskReport:
        intent_risk = self._score_intent(goal + " " + step_description)
        tool_risk = self._score_tool(tool_name)
        profile_risk = self._score_profile()
        environment_risk = 0.05  # Default low environment risk

        composite = (
            intent_risk * WEIGHTS["intent_risk"]
            + tool_risk * WEIGHTS["tool_risk"]
            + profile_risk * WEIGHTS["profile_risk"]
            + environment_risk * WEIGHTS["environment_risk"]
        )
        composite = round(min(composite, 1.0), 4)
        level = self._classify(composite)

        explanation = (
            f"intent={intent_risk:.2f}, tool={tool_risk:.2f}, "
            f"profile={profile_risk:.2f}, env={environment_risk:.2f} "
            f"→ composite={composite:.2f} [{level.value.upper()}]"
        )
        logger.info(f"Risk assessment: {explanation}")

        return RiskReport(
            intent_risk=intent_risk,
            tool_risk=tool_risk,
            profile_risk=profile_risk,
            environment_risk=environment_risk,
            composite_score=composite,
            level=level,
            explanation=explanation,
        )

    def _score_intent(self, text: str) -> float:
        text_lower = text.lower()
        for kw in DANGEROUS_INTENT_KEYWORDS:
            if kw in text_lower:
                return 0.85
        for kw in ELEVATED_INTENT_KEYWORDS:
            if kw in text_lower:
                return 0.45
        return 0.1

    def _score_tool(self, tool_name: Optional[str]) -> float:
        if tool_name is None:
            return 0.0
        return TOOL_RISK_MAP.get(tool_name, TOOL_RISK_MAP["__unknown__"])

    def _score_profile(self) -> float:
        # Higher autonomy level → agent is more trusted → lower profile risk
        profile_risk_by_level = {0: 0.8, 1: 0.5, 2: 0.3, 3: 0.2}
        return profile_risk_by_level.get(self.autonomy_level, 0.5)

    def _classify(self, score: float) -> RiskLevel:
        if score >= THRESHOLDS[RiskLevel.CONFIRM]:
            return RiskLevel.BLOCK
        if score >= THRESHOLDS[RiskLevel.REVIEW]:
            return RiskLevel.CONFIRM
        if score >= THRESHOLDS[RiskLevel.SAFE]:
            return RiskLevel.REVIEW
        return RiskLevel.SAFE

