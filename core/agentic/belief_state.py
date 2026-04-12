"""Persisted belief state for the agentic layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path("data/agentic")
BELIEF_STATE_PATH = DATA_DIR / "belief_state.json"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class BeliefState:
    agent_confidence: float = 0.5
    risk_tolerance: float = 0.5
    user_alignment: float = 0.8

    def __post_init__(self) -> None:
        self.agent_confidence = _clamp(self.agent_confidence)
        self.risk_tolerance = _clamp(self.risk_tolerance)
        self.user_alignment = _clamp(self.user_alignment)

    def to_dict(self) -> dict[str, float]:
        return {key: _clamp(value) for key, value in asdict(self).items()}

    def scores(self) -> dict[str, float]:
        return self.to_dict()

    def update(self, field_name: str, delta: float) -> float:
        current = _clamp(getattr(self, field_name))
        updated = _clamp(current + float(delta))
        setattr(self, field_name, updated)
        return updated

    def should_ask_user(self, action_name: str = "", risk_score: float = 0.0) -> bool:
        del action_name
        return self.agent_confidence < 0.4 or float(risk_score) > self.risk_tolerance

    def save(self) -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        BELIEF_STATE_PATH.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return BELIEF_STATE_PATH

    def load(self) -> "BeliefState":
        if not BELIEF_STATE_PATH.exists():
            return self
        payload = json.loads(BELIEF_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key, value in payload.items():
                if hasattr(self, key):
                    setattr(self, key, _clamp(float(value)))
        return self


__all__ = ["BELIEF_STATE_PATH", "DATA_DIR", "BeliefState"]
