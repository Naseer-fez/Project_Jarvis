"""
core/metrics/risk.py

Computes a numeric risk score (0.0 – 1.0) for a proposed action.

High risk → agent should pause / escalate.
Low risk  → agent may proceed autonomously.

Risk factors:
- Reversibility   : Can the action be undone? (irreversible = high risk)
- Blast radius    : How many resources/users are affected?
- Data sensitivity: Does it touch PII or credentials?
- External effect : Does it leave the system boundary (email, API call)?
- Novelty         : Has the agent done this before?

Usage:
    model = RiskModel()
    model.update("reversibility",   0.9)   # 0=reversible, 1=irreversible
    model.update("blast_radius",    0.3)
    model.update("data_sensitivity",0.5)
    score = model.score()                  # e.g. 0.62
    print(model.verdict())                 # "medium"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RiskLevel(str, Enum):
    LOW      = "low"       # score < 0.3
    MEDIUM   = "medium"    # 0.3 ≤ score < 0.6
    HIGH     = "high"      # 0.6 ≤ score < 0.85
    CRITICAL = "critical"  # score ≥ 0.85


DEFAULT_WEIGHTS: dict[str, float] = {
    "reversibility":    0.35,
    "blast_radius":     0.25,
    "data_sensitivity": 0.20,
    "external_effect":  0.12,
    "novelty":          0.08,
}

THRESHOLDS = {
    RiskLevel.LOW:      (0.0,  0.30),
    RiskLevel.MEDIUM:   (0.30, 0.60),
    RiskLevel.HIGH:     (0.60, 0.85),
    RiskLevel.CRITICAL: (0.85, 1.01),
}


@dataclass
class RiskSignal:
    name: str
    value: float
    weight: float
    updated_at: datetime = field(default_factory=_utcnow)
    justification: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": round(self.weight, 4),
            "updated_at": self.updated_at.isoformat(),
            "justification": self.justification,
        }


class RiskModel:
    """
    Weighted composite risk scorer for a proposed agent action.

    Higher values mean higher risk.  Unset signals default to 0.0 (safe).
    """

    DEFAULT_SIGNAL_VALUE = 0.0   # assume safe until proven otherwise

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._signals: dict[str, RiskSignal] = {}

    # ── Update ───────────────────────────────────────────────────────────

    def update(self, factor: str, value: float, justification: str = "") -> None:
        value = max(0.0, min(1.0, value))
        weight = self._weights.get(factor, 0.05)
        if factor not in self._weights:
            self._weights[factor] = weight
        self._signals[factor] = RiskSignal(
            name=factor,
            value=value,
            weight=weight,
            justification=justification,
        )

    def set_weight(self, factor: str, weight: float) -> None:
        self._weights[factor] = max(0.0, weight)

    # ── Compute ──────────────────────────────────────────────────────────

    def score(self) -> float:
        total_weight = sum(self._weights.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            self._weights[name] * (self._signals[name].value if name in self._signals else self.DEFAULT_SIGNAL_VALUE)
            for name in self._weights
        )
        return round(weighted_sum / total_weight, 4)

    def level(self) -> RiskLevel:
        s = self.score()
        for level, (lo, hi) in THRESHOLDS.items():
            if lo <= s < hi:
                return level
        return RiskLevel.CRITICAL

    def verdict(self) -> str:
        return self.level().value

    def is_safe(self, threshold: float = 0.6) -> bool:
        return self.score() < threshold

    # ── Introspection ────────────────────────────────────────────────────

    def breakdown(self) -> list[dict]:
        rows = []
        for name, weight in self._weights.items():
            sig = self._signals.get(name)
            rows.append({
                "factor": name,
                "value": round(sig.value if sig else self.DEFAULT_SIGNAL_VALUE, 4),
                "weight": round(weight, 4),
                "set": sig is not None,
                "justification": sig.justification if sig else "",
            })
        return sorted(rows, key=lambda r: -r["weight"])

    def snapshot(self) -> dict:
        return {
            "score": self.score(),
            "level": self.verdict(),
            "factors": self.breakdown(),
        }

    def __repr__(self) -> str:
        return f"RiskModel(score={self.score():.3f}, level={self.verdict()})"

