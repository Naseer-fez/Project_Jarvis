"""Persistent belief scores for the Agentic Layer."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/agentic")
BELIEF_STATE_PATH = DATA_DIR / "belief_state.json"
BELIEF_KEYS = (
    "system_reliability",
    "network_reliability",
    "agent_confidence",
    "api_rate_limit_risk",
    "risk_tolerance",
    "user_interruption_likelihood",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(float(value), 4)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@dataclass
class BeliefState:
    """Mutable confidence scores used by autonomy and reflection."""

    system_reliability: float = 0.75
    network_reliability: float = 0.7
    agent_confidence: float = 0.72
    api_rate_limit_risk: float = 0.2
    risk_tolerance: float = 0.5
    user_interruption_likelihood: float = 0.3
    updated_at: str = field(default_factory=_utcnow_iso)

    def scores(self) -> dict[str, float]:
        return {key: _clamp(float(getattr(self, key))) for key in BELIEF_KEYS}

    def update(self, key: str, delta: float) -> float:
        """Adjust a belief score by delta and clamp the result to [0.0, 1.0]."""
        if key not in BELIEF_KEYS:
            raise KeyError(f"Unknown belief key: {key}")

        current_value = float(getattr(self, key))
        new_value = _clamp(current_value + float(delta))
        setattr(self, key, new_value)
        self.updated_at = _utcnow_iso()
        logger.info(
            "Belief updated key=%s old=%.3f delta=%.3f new=%.3f",
            key,
            current_value,
            delta,
            new_value,
        )
        self.save()
        return new_value

    def should_ask_user(self) -> bool:
        """Return True when the agent should seek confirmation before acting."""
        return (
            self.agent_confidence < 0.45
            or self.risk_tolerance < 0.35
            or self.user_interruption_likelihood > 0.7
            or self.api_rate_limit_risk > 0.8
            or not self.is_reliable_enough()
        )

    def is_reliable_enough(self) -> bool:
        """Return True when environment and internal confidence look stable."""
        combined_reliability = (
            self.system_reliability
            + self.network_reliability
            + self.agent_confidence
        ) / 3.0
        return combined_reliability >= 0.6 and self.api_rate_limit_risk <= 0.8

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "updated_at": self.updated_at,
            "scores": self.scores(),
        }

    def save(self) -> Path:
        """Persist the current belief state to disk."""
        _atomic_write_json(BELIEF_STATE_PATH, self.to_dict())
        logger.debug("Belief state saved to %s", BELIEF_STATE_PATH)
        return BELIEF_STATE_PATH

    def load(self) -> BeliefState:
        """Load belief state from disk, keeping defaults if no file exists."""
        if not BELIEF_STATE_PATH.exists():
            logger.info("Belief state file not found at %s; using defaults", BELIEF_STATE_PATH)
            return self

        with BELIEF_STATE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        score_payload = payload
        if isinstance(payload, dict) and "scores" in payload:
            score_payload = payload["scores"]

        if isinstance(score_payload, dict):
            for key in BELIEF_KEYS:
                if key in score_payload:
                    setattr(self, key, _clamp(float(score_payload[key])))

        if isinstance(payload, dict) and payload.get("updated_at"):
            self.updated_at = str(payload["updated_at"])
        else:
            self.updated_at = _utcnow_iso()

        logger.info("Belief state loaded from %s", BELIEF_STATE_PATH)
        return self


__all__ = ["BeliefState", "BELIEF_KEYS", "BELIEF_STATE_PATH"]
