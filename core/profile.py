"""
core/profile.py
════════════════
UserProfileEngine — persistent user identity and behavioral model.
Loads from disk, tracks interaction patterns, updates via synthesis.
"""

import json
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)

PROFILE_PATH = Path("memory/user_profile.json")

DEFAULT_PROFILE = {
    "identity_core": {
        "name": "Unknown",
        "communication_style": "default",
        "expertise_level": "intermediate",
    },
    "preferences": {},
    "behavioral_stats": {
        "intent_counts": {},
        "total_interactions": 0,
        "session_count": 0,
    },
    "confidence_score": 0.0,
    "last_updated": None,
}


class UserProfileEngine:
    """
    Manages the persistent user profile.
    Loaded from disk on init, saved on update.
    """

    def __init__(self, profile_path: str | Path = PROFILE_PATH):
        self.profile_path = Path(profile_path)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = self._load()
        self._session_intents = Counter()

    def _load(self) -> dict:
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"User profile loaded from {self.profile_path}")
                return data
            except Exception as e:
                logger.warning(f"Could not load profile: {e} — using default")
        return json.loads(json.dumps(DEFAULT_PROFILE))  # deep copy

    def _save(self):
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save profile: {e}")

    def log_interaction(self, intent: str):
        """Track interaction for behavioral stats."""
        self._session_intents[intent] += 1
        stats = self.profile.setdefault("behavioral_stats", {})
        counts = stats.setdefault("intent_counts", {})
        counts[intent] = counts.get(intent, 0) + 1
        stats["total_interactions"] = stats.get("total_interactions", 0) + 1
        self._update_confidence()

    def _update_confidence(self):
        """Confidence grows with interactions, capped at 1.0."""
        total = self.profile["behavioral_stats"].get("total_interactions", 0)
        # Sigmoid-like: approaches 1.0 asymptotically
        score = min(1.0, total / (total + 20))
        self.profile["confidence_score"] = round(score, 3)

    def get_confidence_score(self) -> float:
        return self.profile.get("confidence_score", 0.0)

    def get_profile_summary(self) -> str:
        p = self.profile
        ic = p.get("identity_core", {})
        stats = p.get("behavioral_stats", {})
        return (
            f"Name: {ic.get('name', 'Unknown')} | "
            f"Style: {ic.get('communication_style', 'default')} | "
            f"Expertise: {ic.get('expertise_level', 'intermediate')} | "
            f"Interactions: {stats.get('total_interactions', 0)} | "
            f"Confidence: {p.get('confidence_score', 0.0):.2f}"
        )

    def update_profile(self, delta: dict):
        """
        Merge a delta dict into the profile.
        Delta comes from ProfileSynthesizer.
        """
        def deep_merge(base: dict, updates: dict):
            for k, v in updates.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v

        deep_merge(self.profile, delta)
        self.profile["last_updated"] = datetime.utcnow().isoformat()
        self._save()
        logger.info("User profile updated and saved.")
