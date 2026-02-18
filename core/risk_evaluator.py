"""
core/risk_evaluator.py
═══════════════════════
Stateless, deterministic, table-based risk scoring.

V1 Rules:
  - No ML. No probability models. Simple table lookup.
  - Physical actions are HARD BLOCKED — not just high risk
  - If a tool is unknown → blocked by default (fail safe)
  - All evaluations are logged
  - RiskEvaluator is the LAST LINE of defense before any tool call

Risk Levels:
  0.0 - 0.3  → SAFE  (proceed)
  0.3 - 0.6  → CAUTION (log, proceed with note)
  0.6 - 0.9  → HIGH (require explicit confirmation flag)
  0.9 - 1.0  → CRITICAL (block)
  BLOCKED     → Hard block, never executable in V1
"""

from dataclasses import dataclass
from enum import Enum
from core.logger import get_logger, audit

logger = get_logger("risk_evaluator")


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    BLOCKED = "BLOCKED"  # Hardcoded NEVER in V1


@dataclass(frozen=True)
class RiskResult:
    tool: str
    score: float
    level: RiskLevel
    allowed: bool
    reason: str


# ══════════════════════════════════════════════
# RISK TABLE — Only this file changes risk scores
# ══════════════════════════════════════════════
RISK_TABLE: dict[str, float | str] = {
    # Perception tools (L0)
    "vision.analyze_image":        0.1,
    "vision.snapshot":             0.1,
    "memory.read":                 0.1,

    # Planning/Reasoning tools (L1)
    "memory.write":                0.2,
    "planner.generate_plan":       0.2,
    "risk_evaluator.evaluate":     0.1,

    # Digital interaction tools (L2 — V2+)
    "filesystem.read":             0.4,
    "filesystem.write":            0.6,
    "tts.speak":                   0.2,

    # Physical / System tools — HARD BLOCKED IN V1
    "desktop_automation.click":    "BLOCKED",
    "desktop_automation.type":     "BLOCKED",
    "desktop_automation.scroll":   "BLOCKED",
    "serial_controller.send":      "BLOCKED",
    "serial_controller.move":      "BLOCKED",
    "shell.execute":               "BLOCKED",
}

# Thresholds
THRESHOLD_CAUTION = 0.3
THRESHOLD_HIGH = 0.6
THRESHOLD_CRITICAL = 0.9


class RiskEvaluator:
    """
    Stateless risk evaluator. Call evaluate() before every tool call.
    If it returns allowed=False, the call MUST NOT proceed.
    """

    def evaluate(self, tool_name: str, context: dict | None = None) -> RiskResult:
        """
        Evaluate risk for a given tool.
        Always returns a RiskResult. Never raises.
        """
        raw = RISK_TABLE.get(tool_name)

        # Unknown tool → block by default (fail safe)
        if raw is None:
            result = RiskResult(
                tool=tool_name,
                score=1.0,
                level=RiskLevel.BLOCKED,
                allowed=False,
                reason=f"Unknown tool '{tool_name}' — blocked by default (fail-safe policy)"
            )
            self._log(result)
            return result

        # Hard block
        if raw == "BLOCKED":
            result = RiskResult(
                tool=tool_name,
                score=1.0,
                level=RiskLevel.BLOCKED,
                allowed=False,
                reason=f"Tool '{tool_name}' is HARD BLOCKED in V1 (physical/system action)"
            )
            self._log(result)
            return result

        score: float = raw
        level = self._score_to_level(score)
        allowed = level not in (RiskLevel.CRITICAL, RiskLevel.BLOCKED)

        result = RiskResult(
            tool=tool_name,
            score=score,
            level=level,
            allowed=allowed,
            reason=self._reason(tool_name, score, level)
        )
        self._log(result)
        return result

    def evaluate_plan(self, steps: list[dict]) -> tuple[bool, list[RiskResult]]:
        """
        Evaluate all steps in a plan.
        Returns (all_safe, results_list).
        Plan is safe only if ALL steps are allowed.
        """
        results = []
        for step in steps:
            tool = step.get("action", "unknown")
            result = self.evaluate(tool)
            results.append(result)

        all_safe = all(r.allowed for r in results)
        if not all_safe:
            blocked = [r for r in results if not r.allowed]
            logger.warning(
                f"PLAN BLOCKED: {len(blocked)}/{len(results)} steps failed risk check. "
                f"Blocked tools: {[r.tool for r in blocked]}"
            )
        return all_safe, results

    def _score_to_level(self, score: float) -> RiskLevel:
        if score >= THRESHOLD_CRITICAL:
            return RiskLevel.CRITICAL
        elif score >= THRESHOLD_HIGH:
            return RiskLevel.HIGH
        elif score >= THRESHOLD_CAUTION:
            return RiskLevel.CAUTION
        else:
            return RiskLevel.SAFE

    def _reason(self, tool: str, score: float, level: RiskLevel) -> str:
        reasons = {
            RiskLevel.SAFE: f"Tool '{tool}' is low-risk (score={score}). Proceed.",
            RiskLevel.CAUTION: f"Tool '{tool}' has moderate risk (score={score}). Logging.",
            RiskLevel.HIGH: f"Tool '{tool}' is high-risk (score={score}). Confirmation recommended.",
            RiskLevel.CRITICAL: f"Tool '{tool}' exceeds critical threshold (score={score}). BLOCKED.",
        }
        return reasons.get(level, "Unknown risk level.")

    def _log(self, result: RiskResult):
        level_fn = logger.warning if not result.allowed else logger.debug
        level_fn(
            f"RISK [{result.level.value}] tool={result.tool} score={result.score} "
            f"allowed={result.allowed} | {result.reason}"
        )
        audit(
            logger,
            f"RISK_EVAL: tool={result.tool} level={result.level.value} allowed={result.allowed}",
            tool=result.tool,
            risk=result.score,
        )
