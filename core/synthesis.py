"""Profile synthesis engine for Session 3."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from core.profile import UserProfileEngine


SYNTHESIS_SYSTEM = """You are analyzing conversation history to understand a user's
communication style and expertise level. Respond ONLY with a JSON object.
No preamble. No markdown. No code fences. Raw JSON only.
Format:
{
  "communication_style": {"value": "casual|formal|technical", "confidence": 0.0-1.0},
  "expertise_level": {"value": "beginner|intermediate|advanced|expert", "confidence": 0.0-1.0},
  "preferred_topics": {"value": ["topic1", "topic2"], "confidence": 0.0-1.0},
  "name": {"value": "FirstName", "confidence": 0.0-1.0}
}
Only include fields you are confident about. Omit fields you cannot determine.
"""


class ProfileSynthesizer:
    def __init__(self, llm):
        self._llm = llm

    def should_run(self, profile: UserProfileEngine) -> bool:
        count = profile.interaction_count
        return count > 0 and count % 20 == 0

    async def synthesize(self, recent_conversations: list, profile: UserProfileEngine) -> dict:
        """Analyze recent conversations and apply a confident profile delta."""
        if not recent_conversations:
            return {"updated_fields": [], "delta": {}}

        convo_text = "\n\n".join(str(item) for item in recent_conversations[-20:])
        prompt = f"Analyze these conversations and extract user profile signals:\n\n{convo_text}"

        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._call_llm(prompt),
            )
            clean = self._strip_fences(raw)
            delta = json.loads(clean)
            updated = profile.apply_delta(delta)
            return {"updated_fields": updated, "delta": delta}
        except json.JSONDecodeError as e:
            logging.getLogger(__name__).warning(f"Synthesis JSON parse failed: {e}")
            return {"updated_fields": [], "delta": {}, "error": "json_parse_failed"}
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning(f"Synthesis failed: {e}")
            return {"updated_fields": [], "delta": {}, "error": str(e)}

    def _call_llm(self, prompt: str) -> str:
        if self._llm is None:
            raise RuntimeError("LLM unavailable for profile synthesis.")

        if hasattr(self._llm, "complete"):
            try:
                result = self._llm.complete(prompt, system=SYNTHESIS_SYSTEM, task_type="synthesis")
            except TypeError:
                result = self._llm.complete(prompt, system=SYNTHESIS_SYSTEM)
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return str(result or "")

        if hasattr(self._llm, "chat"):
            result = self._llm.chat(prompt, system_context=SYNTHESIS_SYSTEM)
            return str(result or "")

        raise AttributeError("LLM client does not expose complete() or chat().")

    @staticmethod
    def _strip_fences(text: str) -> str:
        clean = (text or "").strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
        clean = clean.strip()

        if clean.startswith("{") and clean.endswith("}"):
            return clean

        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            return clean[start : end + 1]

        return clean
