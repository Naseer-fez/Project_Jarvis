"""
core/intelligence.py
═════════════════════
MemoryIntelligence — decides whether to passively store facts from conversation.
Uses LLM to extract key/value pairs and categorize them.
"""

import logging
import re

logger = logging.getLogger(__name__)

MEMORY_EVAL_SYSTEM = """You are a memory extraction module for a personal AI assistant.

Analyze the user message and decide if it contains a storable fact, preference, or episode.

Rules:
- Preferences: things the user likes/dislikes/wants (e.g. "I prefer dark mode")
- Facts: explicit statements about themselves (e.g. "My name is Alex, I'm 30")
- Episodes: notable events (e.g. "I just finished my project")
- If nothing worth storing → decision = "ignore"

Respond ONLY with valid JSON:
{"decision": "store"|"ignore", "key": "short_key", "value": "what to store", "category": "preference"|"fact"|"episode"}

If ignoring: {"decision": "ignore"}"""


class MemoryIntelligence:
    """
    Passively evaluates conversation input and decides what to remember.
    Does NOT store directly — returns a decision dict for the controller.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def evaluate_input(self, user_input: str) -> dict | None:
        """
        Returns a storage decision dict or None.
        {"decision": "store", "key": ..., "value": ..., "category": ...}
        {"decision": "ignore"} or None
        """
        # Quick heuristic pre-filter — skip short messages
        if len(user_input.strip()) < 15:
            return None

        if self.llm:
            result = self._llm_evaluate(user_input)
            if result:
                return result

        return self._heuristic_evaluate(user_input)

    def _llm_evaluate(self, text: str) -> dict | None:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self.llm.complete_json(
                    f"Should I remember this?\n\n{text}",
                    system=MEMORY_EVAL_SYSTEM,
                    temperature=0.0
                )
            )
            loop.close()
            if result and result.get("decision") == "store":
                return result
            return {"decision": "ignore"}
        except Exception as e:
            logger.warning(f"MemoryIntelligence LLM failed: {e}")
            return None

    def _heuristic_evaluate(self, text: str) -> dict | None:
        """Keyword-based fallback for memory extraction."""
        lower = text.lower()

        preference_signals = ["i prefer", "i like", "i love", "i hate", "i always", "i never", "my favourite"]
        fact_signals = ["my name is", "i am", "i'm", "i work", "i live", "i was born"]

        for sig in preference_signals:
            if sig in lower:
                return {
                    "decision": "store",
                    "key": "preference",
                    "value": text.strip(),
                    "category": "preference"
                }

        for sig in fact_signals:
            if sig in lower:
                return {
                    "decision": "store",
                    "key": "personal_fact",
                    "value": text.strip(),
                    "category": "fact"
                }

        return {"decision": "ignore"}
