"""
belief_state.py — Agent Belief State for Jarvis Agentic Layer

Stores the agent's evolving beliefs about its environment, tools, and itself.
Beliefs are probabilities / confidence scores in [0.0, 1.0].
They influence future planning decisions without hard-coding behavior.

These are beliefs, not facts.  They are updated by the ReflectionEngine.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BELIEF_FILE = Path("data/agentic/belief_state.json")

# Default beliefs on first boot
_DEFAULTS: Dict[str, float] = {
    "system_reliability": 0.80,     # Confidence the overall system works
    "network_reliability": 0.85,    # Confidence network calls will succeed
    "agent_confidence": 0.70,       # Agent's self-assessed competence
    "api_rate_limit_risk": 0.10,    # Estimated risk of hitting rate limits
    "user_interruption_likelihood": 0.20,  # How often user wants manual control
    "task_complexity_bias": 0.50,   # Tendency to underestimate task complexity
    "risk_tolerance": 0.40,         # Current tolerance for autonomous action
}

# Caps
_MIN = 0.0
_MAX = 1.0


class BeliefEntry:
    """One belief with its history."""

    def __init__(self, key: str, value: float):
        self.key = key
        self.value: float = max(_MIN, min(_MAX, value))
        self.history: List[Tuple[str, float, str]] = []  # (timestamp, value, source)

    def update(self, delta: float, source: str = "unknown") -> float:
        old = self.value
        self.value = max(_MIN, min(_MAX, self.value + delta))
        self.history.append((datetime.utcnow().isoformat(), self.value, source))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        logger.debug("Belief '%s': %.2f → %.2f (Δ%+.2f via %s)", self.key, old, self.value, delta, source)
        return self.value

    def set(self, value: float, source: str = "direct") -> float:
        self.history.append((datetime.utcnow().isoformat(), self.value, source))
        self.value = max(_MIN, min(_MAX, value))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        return self.value

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "value": self.value,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BeliefEntry":
        entry = cls(data["key"], data["value"])
        entry.history = data.get("history", [])
        return entry


class BeliefState:
    """
    Persistent agent belief state.

    Usage:
        bs = BeliefState()
        bs.load()
        bs.update("network_reliability", -0.1, source="reflection")
        print(bs.get("network_reliability"))
        bs.save()
    """

    def __init__(self, storage_path: Path = BELIEF_FILE):
        self.storage_path = storage_path
        self._beliefs: Dict[str, BeliefEntry] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        for key, value in _DEFAULTS.items():
            if key not in self._beliefs:
                self._beliefs[key] = BeliefEntry(key, value)

    # ---------------------------------------------------------------- I/O

    def load(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            logger.info("No belief file found — using defaults.")
            return
        try:
            with self.storage_path.open() as f:
                raw = json.load(f)
            self._beliefs = {k: BeliefEntry.from_dict(v) for k, v in raw.items()}
            self._init_defaults()  # ensure new keys are always present
            logger.info("Belief state loaded (%d beliefs).", len(self._beliefs))
        except Exception as exc:
            logger.error("Failed to load belief state: %s", exc)

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.storage_path.with_suffix(".tmp")
        try:
            with tmp.open("w") as f:
                json.dump({k: v.to_dict() for k, v in self._beliefs.items()}, f, indent=2)
            tmp.replace(self.storage_path)
        except Exception as exc:
            logger.error("Failed to save belief state: %s", exc)

    # ---------------------------------------------------------- Access

    def get(self, key: str, default: float = 0.5) -> float:
        """Return current belief value, or default if unknown."""
        entry = self._beliefs.get(key)
        return entry.value if entry else default

    def update(self, key: str, delta: float, source: str = "unknown") -> float:
        """Nudge a belief by delta. Creates it at 0.5 if it doesn't exist."""
        if key not in self._beliefs:
            self._beliefs[key] = BeliefEntry(key, 0.5)
        value = self._beliefs[key].update(delta, source)
        self.save()
        return value

    def set_belief(self, key: str, value: float, source: str = "direct") -> float:
        """Directly set a belief (use sparingly — prefer update())."""
        if key not in self._beliefs:
            self._beliefs[key] = BeliefEntry(key, value)
        result = self._beliefs[key].set(value, source)
        self.save()
        return result

    def snapshot(self) -> Dict[str, float]:
        """Return a plain dict of all current belief values."""
        return {k: v.value for k, v in self._beliefs.items()}

    def history(self, key: str) -> List[Tuple[str, float, str]]:
        """Return change history for a belief key."""
        entry = self._beliefs.get(key)
        return entry.history if entry else []

    # -------------------------------------------------------- Derived helpers

    def is_reliable_enough(self, threshold: float = 0.6) -> bool:
        """True if system_reliability belief is above threshold."""
        return self.get("system_reliability") >= threshold

    def should_ask_user(self) -> bool:
        """
        True if risk_tolerance is low or agent_confidence is low.
        Used by AutonomyPolicy as one signal.
        """
        return (
            self.get("risk_tolerance") < 0.35
            or self.get("agent_confidence") < 0.40
        )

    def summary_string(self) -> str:
        lines = ["Current Beliefs:"]
        for key, entry in sorted(self._beliefs.items()):
            bar = "█" * int(entry.value * 10) + "░" * (10 - int(entry.value * 10))
        lines.append(f"  {key:<35} {bar} {entry.value:.2f}")
        return "\n".join(lines)
