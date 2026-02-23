"""
core/metrics/decay.py

Time-based decay functions for confidence and risk scores.

Confidence decays over time if nothing is updated (stale knowledge).
Risk decays after a safe period with no incidents.

Design: pure functions + a DecaySchedule helper. No background threads.

Usage:
    score = exponential_decay(current=0.9, elapsed_seconds=3600, half_life=1800)
    # → ~0.45 after one half-life

    schedule = DecaySchedule(initial=0.8, half_life_seconds=600)
    score = schedule.value_at(elapsed_seconds=300)  # halfway through
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Pure decay functions ──────────────────────────────────────────────────────

def exponential_decay(
    current: float,
    elapsed_seconds: float,
    half_life: float,
    floor: float = 0.0,
) -> float:
    """
    Standard exponential decay.

    value(t) = current * 0.5 ^ (elapsed / half_life)

    Args:
        current:         Value at t=0 (should be in [0, 1]).
        elapsed_seconds: How many seconds have passed since t=0.
        half_life:       Seconds for value to halve.
        floor:           Minimum value (never decays below this).
    """
    if half_life <= 0 or elapsed_seconds < 0:
        return max(floor, current)
    factor = 0.5 ** (elapsed_seconds / half_life)
    return max(floor, current * factor)


def linear_decay(
    current: float,
    elapsed_seconds: float,
    decay_per_second: float,
    floor: float = 0.0,
) -> float:
    """
    Linear decay: decreases by a fixed amount per second.
    """
    decayed = current - decay_per_second * elapsed_seconds
    return max(floor, decayed)


def step_decay(
    current: float,
    elapsed_seconds: float,
    step_interval: float,
    step_amount: float,
    floor: float = 0.0,
) -> float:
    """
    Staircase decay: drops by step_amount every step_interval seconds.
    """
    steps = int(elapsed_seconds // step_interval)
    decayed = current - steps * step_amount
    return max(floor, decayed)


# ── DecaySchedule ─────────────────────────────────────────────────────────────

@dataclass
class DecaySchedule:
    """
    Applies a chosen decay function to a score starting from a known time.

    Usage:
        sched = DecaySchedule(initial=0.85, half_life_seconds=1800)
        current_score = sched.current_value()   # call any time
    """

    initial: float
    half_life_seconds: float = 1800.0       # 30 minutes default
    floor: float = 0.1
    ceiling: float = 1.0
    started_at: datetime = None             # type: ignore[assignment]
    strategy: str = "exponential"           # "exponential" | "linear" | "step"

    # For linear strategy
    linear_decay_per_second: float = 0.0001

    # For step strategy
    step_interval: float = 300.0
    step_amount: float = 0.05

    def __post_init__(self) -> None:
        if self.started_at is None:
            self.started_at = _utcnow()

    def elapsed_seconds(self, at: Optional[datetime] = None) -> float:
        at = at or _utcnow()
        return max(0.0, (at - self.started_at).total_seconds())

    def value_at(self, elapsed_seconds: Optional[float] = None, at: Optional[datetime] = None) -> float:
        if elapsed_seconds is None:
            elapsed_seconds = self.elapsed_seconds(at)

        if self.strategy == "exponential":
            v = exponential_decay(self.initial, elapsed_seconds, self.half_life_seconds, self.floor)
        elif self.strategy == "linear":
            v = linear_decay(self.initial, elapsed_seconds, self.linear_decay_per_second, self.floor)
        elif self.strategy == "step":
            v = step_decay(self.initial, elapsed_seconds, self.step_interval, self.step_amount, self.floor)
        else:
            v = self.initial

        return min(self.ceiling, max(self.floor, round(v, 4)))

    def current_value(self) -> float:
        return self.value_at()

    def is_expired(self, threshold: float) -> bool:
        """True when decayed value has dropped below threshold."""
        return self.current_value() < threshold

    def time_until_threshold(self, threshold: float) -> Optional[float]:
        """
        Seconds until value decays to threshold (exponential only).
        Returns None if already below threshold or not exponential.
        """
        if self.strategy != "exponential":
            return None
        if self.initial <= threshold:
            return 0.0
        # t = half_life * log2(initial / threshold)
        ratio = self.initial / max(threshold, 1e-9)
        seconds = self.half_life_seconds * math.log2(ratio)
        already_elapsed = self.elapsed_seconds()
        remaining = seconds - already_elapsed
        return max(0.0, remaining) if remaining > 0 else None

    def reset(self, new_initial: Optional[float] = None) -> None:
        """Restart the decay clock, optionally with a new initial value."""
        self.started_at = _utcnow()
        if new_initial is not None:
            self.initial = max(self.floor, min(self.ceiling, new_initial))

    def __repr__(self) -> str:
        return (
            f"DecaySchedule(initial={self.initial}, "
            f"current={self.current_value():.3f}, "
            f"strategy={self.strategy})"
        )

