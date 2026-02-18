"""
core/synthesis.py
══════════════════
ProfileSynthesizer — uses LLM to synthesize a user identity delta
from accumulated memories and interaction history.

Called by controller after N new memories or on shutdown.
Returns a delta dict that UserProfileEngine merges into the profile.
"""

import logging
import json

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """You are a user modelling system for a personal AI assistant called Jarvis.

Given the user's stored memories and interaction history, synthesize an updated identity profile delta.

Focus on:
- communication_style: formal | casual | technical | creative
- expertise_level: beginner | intermediate | advanced | expert
- name: if mentioned anywhere
- Any strong preferences or personality traits

Respond ONLY with a valid JSON delta (only fields you're confident about):
{
  "identity_core": {
    "name": "...",
    "communication_style": "...",
    "expertise_level": "..."
  }
}

If you cannot determine something with confidence, omit it entirely."""


class ProfileSynthesizer:
    """
    Analyzes stored memories and generates a profile update delta.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def synthesize(self, hybrid_memory) -> dict | None:
        """
        Pull recent memory, ask LLM to synthesize a profile delta.
        Returns delta dict or None if nothing to update.
        """
        # Gather context
        episodes = hybrid_memory.get_recent_episodes(limit=20)
        preferences = hybrid_memory.get_all_preferences()

        if not episodes and not preferences:
            logger.info("Synthesis: no memory to analyze.")
            return None

        context_parts = []
        if preferences:
            context_parts.append("USER PREFERENCES:")
            for k, v in preferences.items():
                context_parts.append(f"  - {k}: {v}")

        if episodes:
            context_parts.append("\nRECENT EPISODES:")
            for ep in episodes[:10]:
                context_parts.append(f"  - {ep['content']}")

        context = "\n".join(context_parts)
        prompt = f"Synthesize a user profile delta from this data:\n\n{context}"

        if self.llm is None:
            logger.warning("Synthesis: no LLM available, returning empty delta.")
            return {}

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            delta = loop.run_until_complete(
                self.llm.complete_json(prompt, system=SYNTHESIS_SYSTEM, temperature=0.2)
            )
            loop.close()

            if delta and isinstance(delta, dict):
                logger.info(f"Synthesis delta: {list(delta.keys())}")
                return delta
            return None

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return None
