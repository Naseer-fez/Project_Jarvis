"""Persistent user profile engine for Session 3 personalization."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path


class UserProfileEngine:
    PROFILE_PATH = Path("memory/user_profile.json")

    DEFAULTS = {
        "name": "User",
        "communication_style": "casual",
        "expertise_level": "intermediate",
        "preferred_topics": [],
        "timezone": "UTC",
        "language": "en",
        "interaction_count": 0,
        "first_seen": None,
        "last_seen": None,
    }

    _VALID_STYLES = {"casual", "formal", "technical"}
    _VALID_LEVELS = {"beginner", "intermediate", "advanced", "expert"}

    def __init__(self) -> None:
        self._data = self._fresh_defaults()
        self._load()

    def _fresh_defaults(self) -> dict:
        data = dict(self.DEFAULTS)
        data["preferred_topics"] = []
        return data

    def _load(self) -> None:
        try:
            if self.PROFILE_PATH.exists():
                with open(self.PROFILE_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if k in self._data:
                            self._data[k] = v

            if not isinstance(self._data.get("preferred_topics"), list):
                self._data["preferred_topics"] = []
            if self._data.get("communication_style") not in self._VALID_STYLES:
                self._data["communication_style"] = self.DEFAULTS["communication_style"]
            if self._data.get("expertise_level") not in self._VALID_LEVELS:
                self._data["expertise_level"] = self.DEFAULTS["expertise_level"]
            if not isinstance(self._data.get("interaction_count"), int):
                self._data["interaction_count"] = int(self._data.get("interaction_count") or 0)
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning(f"Profile load failed: {e}")
            self._data = self._fresh_defaults()

    def save(self) -> None:
        """Atomic write to avoid corruption on interruption."""
        self.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.PROFILE_PATH.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.PROFILE_PATH)
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error(f"Profile save failed: {e}")

    def update_from_conversation(self, user_text: str, jarvis_response: str) -> None:
        now = datetime.now().isoformat()
        self._data["interaction_count"] = int(self._data.get("interaction_count", 0)) + 1
        self._data["last_seen"] = now
        if self._data.get("first_seen") is None:
            self._data["first_seen"] = now

        lower = (user_text or "").lower()
        for pattern in ("my name is ", "i am ", "i'm ", "call me "):
            if pattern in lower:
                idx = lower.index(pattern) + len(pattern)
                remainder = (user_text or "")[idx:].strip()
                if not remainder:
                    break
                candidate = remainder.split()[0].strip(".,!?\"'()[]{}")
                if 2 <= len(candidate) <= 30 and candidate.isalpha():
                    self._data["name"] = candidate
                    break

        _ = jarvis_response
        self.save()

    def apply_delta(self, delta: dict, min_confidence: float = 0.6) -> list:
        """Apply synthesis delta and return list of updated fields."""
        if not isinstance(delta, dict):
            return []

        updated: list[str] = []
        for field, val in delta.items():
            if isinstance(val, dict):
                try:
                    confidence = float(val.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                value = val.get("value")
            else:
                confidence = 1.0
                value = val

            if confidence < min_confidence or field not in self.DEFAULTS or value is None:
                continue

            if field == "communication_style":
                if value not in self._VALID_STYLES:
                    continue
            elif field == "expertise_level":
                if value not in self._VALID_LEVELS:
                    continue
            elif field == "preferred_topics":
                if not isinstance(value, list):
                    continue
                cleaned_topics = []
                for topic in value:
                    topic_text = str(topic).strip()
                    if topic_text and topic_text not in cleaned_topics:
                        cleaned_topics.append(topic_text)
                    if len(cleaned_topics) >= 10:
                        break
                value = cleaned_topics
            elif field == "name":
                text = str(value).strip()
                if not text or len(text) > 30:
                    continue
                value = text

            self._data[field] = value
            updated.append(field)

        if updated:
            self.save()
        return updated

    def get_system_prompt_injection(self) -> str:
        """Compact profile context injected into the LLM system prompt."""
        parts = [f"User: {self._data['name']}."]
        parts.append(f"Style: {self._data['communication_style']}.")
        parts.append(f"Level: {self._data['expertise_level']}.")
        if self._data.get("preferred_topics"):
            topics = ", ".join(self._data["preferred_topics"][:3])
            parts.append(f"Interests: {topics}.")
        prompt = " ".join(parts)
        words = prompt.split()
        if len(words) > 80:
            return " ".join(words[:80])
        return prompt

    def get_communication_style(self) -> str:
        style = self._data.get("communication_style", "casual")
        return {
            "formal": "Be precise and professional. Use formal language.",
            "casual": "Be friendly and conversational. Keep it natural.",
            "technical": "Be detailed and technical. Use correct terminology.",
        }.get(style, "Be helpful and clear.")

    @property
    def interaction_count(self) -> int:
        return int(self._data.get("interaction_count", 0))
