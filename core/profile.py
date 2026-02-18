"""
core/profile.py
───────────────
User Profile Engine (Session 7).
Aggregates preferences, behaviors, and traits into a structured adaptive profile.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = {
    "identity_core": {
        "name": None,
        "occupation": None,
        "primary_interests": [],
        "communication_style": "unknown"  # concise | detailed | casual | analytical
    },
    "behavioral_patterns": {
        "common_topics": [],
        "frequent_intents": {"CHAT": 0, "COMMAND": 0, "QUERY_MEMORY": 0},
        "interaction_time_patterns": []
    },
    "preference_weights": {
        "tone_preference": 0.5,    # 0.0 (Formal) <-> 1.0 (Casual)
        "detail_level": 0.5,       # 0.0 (Brief)  <-> 1.0 (Comprehensive)
        "risk_tolerance": 0.5      # 0.0 (Safe)   <-> 1.0 (Experimental)
    },
    "confidence_score": 0.0,
    "last_updated": None
}

class UserProfileEngine:
    def __init__(self, profile_path: str = "memory/user_profile.json"):
        self.profile_path = Path(profile_path)
        self.profile = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        """Load existing profile or initialize a new one."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f:
                    data = json.load(f)
                # Merge with default to ensure all schema fields exist
                return self._merge_defaults(data, DEFAULT_PROFILE)
            except Exception as e:
                logger.error(f"Failed to load user profile: {e}")
                return DEFAULT_PROFILE.copy()
        return DEFAULT_PROFILE.copy()

    def _merge_defaults(self, current: dict, default: dict) -> dict:
        """Recursively ensure all keys in default exist in current."""
        for k, v in default.items():
            if k not in current:
                current[k] = v
            elif isinstance(v, dict) and isinstance(current[k], dict):
                self._merge_defaults(current[k], v)
        return current

    def save_profile(self):
        """Persist the current profile to disk."""
        self.profile["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.profile_path, "w") as f:
                json.dump(self.profile, f, indent=2)
            logger.info("User profile saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save user profile: {e}")

    def update_profile(self, memory_delta: Dict[str, Any]):
        """
        Apply a synthesis delta to the profile.
        Requirements: memory_delta must match profile structure.
        """
        # 1. Update Identity Core (only if value provided)
        if "identity_core" in memory_delta:
            for k, v in memory_delta["identity_core"].items():
                if v is not None:
                    self.profile["identity_core"][k] = v

        # 2. Update Behavioral Patterns
        if "behavioral_patterns" in memory_delta:
            bp = memory_delta["behavioral_patterns"]
            if "common_topics" in bp:
                # Merge and dedup
                current = set(self.profile["behavioral_patterns"]["common_topics"])
                current.update(bp["common_topics"])
                self.profile["behavioral_patterns"]["common_topics"] = list(current)[:10]
            
            # Intents are additive
            if "frequent_intents" in bp:
                for intent, count in bp["frequent_intents"].items():
                    current_count = self.profile["behavioral_patterns"]["frequent_intents"].get(intent, 0)
                    self.profile["behavioral_patterns"]["frequent_intents"][intent] = current_count + count

        # 3. Update Preference Weights
        if "preference_weights" in memory_delta:
            self.profile["preference_weights"].update(memory_delta["preference_weights"])

        # 4. Update Confidence
        if "confidence_score" in memory_delta:
            self.profile["confidence_score"] = memory_delta["confidence_score"]

        self.save_profile()

    def get_profile_summary(self) -> str:
        """
        Generate a concise string representation for LLM system prompt injection.
        """
        core = self.profile["identity_core"]
        weights = self.profile["preference_weights"]
        
        summary_parts = ["[USER IDENTITY]"]
        
        if core["name"]:
            summary_parts.append(f"Name: {core['name']}")
        if core["occupation"]:
            summary_parts.append(f"Occupation: {core['occupation']}")
        if core["primary_interests"]:
            summary_parts.append(f"Interests: {', '.join(core['primary_interests'])}")
        
        # Derived Communication Instructions
        style = core.get("communication_style", "unknown")
        summary_parts.append(f"Communication Preference: {style.upper()}")
        
        # Tone guidance based on weights
        tone_instr = []
        if weights["tone_preference"] > 0.7: tone_instr.append("Casual/Friendly")
        elif weights["tone_preference"] < 0.3: tone_instr.append("Formal/Professional")
        
        if weights["detail_level"] > 0.7: tone_instr.append("Comprehensive/Detailed")
        elif weights["detail_level"] < 0.3: tone_instr.append("Concise/Brief")
        
        if tone_instr:
            summary_parts.append(f"Response Tone: {', '.join(tone_instr)}")
            
        return "\n".join(summary_parts)

    def log_interaction(self, intent: str):
        """Update rudimentary stats in real-time."""
        intent = intent.upper()
        stats = self.profile["behavioral_patterns"]["frequent_intents"]
        if intent in stats:
            stats[intent] += 1
