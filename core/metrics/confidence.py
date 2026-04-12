"""Tiny confidence aggregator used by the agent loop."""

from __future__ import annotations


class ConfidenceModel:
    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    def update(self, metric: str, value: float) -> float:
        clamped = max(0.0, min(1.0, float(value)))
        self._scores[str(metric)] = clamped
        return clamped

    def score(self) -> float:
        if not self._scores:
            return 0.5
        return sum(self._scores.values()) / len(self._scores)


__all__ = ["ConfidenceModel"]
