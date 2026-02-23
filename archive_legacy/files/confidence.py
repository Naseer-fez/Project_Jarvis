"""
core/metrics/confidence.py

Computes a numeric confidence score (0.0 – 1.0) that the agent can use
to decide whether to act autonomously or pause for human input.

Design:
- Confidence is a *weighted composite* of multiple signals.
- Each signal is a float in [0, 1].
- Weights are normalised so they always sum to 1.
- The final score is the weighted average.

Usage:
    model = ConfidenceModel()
    model.update("intent_clarity",    0.9)
    model.update("tool_reliability",  0.7)
    model.update("data_freshness",    0.85)
    score = model.score()   # e.g. 0.817
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Default signal weights.  Override via ConfidenceModel(weights={...}).
DEFAULT_WEIGHTS: dict[str, float] = {
    "intent_clarity":   0.30,   # how clear is the user's request?
    "tool_reliability": 0.25,   # how reliable is the chosen tool?
    "data_freshness":   0.20,   # how recent is the underlying data?
    "context_match":    0.15,   # does current context match expected?
    "past_success":     0.10,   # historical success rate for this action
}


@dataclass
class SignalEntry:
    name: str
    value: float                    # 0.0 – 1.0
    weight: float
    updated_at: datetime = field(default_factory=_utcnow)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": round(self.weight, 4),
            "updated_at": self.updated_at.isoformat(),
            "note": self.note,
        }


class ConfidenceModel:
    """
    Weighted composite confidence scorer.

    Signals not yet set default to 0.5 (uncertain).
    """

    DEFAULT_SIGNAL_VALUE = 0.5

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._signals: dict[str, SignalEntry] = {}

    # ── Update ───────────────────────────────────────────────────────────

    def update(self, signal: str, value: float, note: str = "") -> None:
        """Set or update a signal value (clamped to [0, 1])."""
        value = max(0.0, min(1.0, value))
        weight = self._weights.get(signal, 0.05)   # unknown signals get low weight
        if signal not in self._weights:
            self._weights[signal] = weight          # register dynamically
        self._signals[signal] = SignalEntry(
            name=signal,
            value=value,
            weight=weight,
            note=note,
        )

    def set_weight(self, signal: str, weight: float) -> None:
        self._weights[signal] = max(0.0, weight)

    # ── Compute ──────────────────────────────────────────────────────────

    def score(self) -> float:
        """
        Return weighted average confidence across all known signals.
        Signals not yet updated contribute DEFAULT_SIGNAL_VALUE.
        """
        total_weight = sum(self._weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for name, weight in self._weights.items():
            if name in self._signals:
                value = self._signals[name].value
            else:
                value = self.DEFAULT_SIGNAL_VALUE
            weighted_sum += weight * value

        return round(weighted_sum / total_weight, 4)

    def is_confident(self, threshold: float = 0.7) -> bool:
        return self.score() >= threshold

    # ── Introspection ────────────────────────────────────────────────────

    def breakdown(self) -> list[dict]:
        rows = []
        for name, weight in self._weights.items():
            entry = self._signals.get(name)
            rows.append({
                "signal": name,
                "value": round(entry.value if entry else self.DEFAULT_SIGNAL_VALUE, 4),
                "weight": round(weight, 4),
                "set": entry is not None,
                "note": entry.note if entry else "",
            })
        return sorted(rows, key=lambda r: -r["weight"])

    def snapshot(self) -> dict:
        return {
            "score": self.score(),
            "signals": self.breakdown(),
        }

    def __repr__(self) -> str:
        return f"ConfidenceModel(score={self.score():.3f})"

