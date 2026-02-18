"""
core/synthesis.py
─────────────────
Profile Synthesizer (Session 7).
Analyzes memory snapshots to derive stable user traits and patterns.
"""

import json
import logging
from typing import Dict, Any, List

from core.llm import LLMClientV2
from core.profile import DEFAULT_PROFILE

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """
You are the Identity Synthesizer for Jarvis.
Analyze the following MEMORY SNAPSHOT and derive a structured user profile.

MEMORY SNAPSHOT:
{memory_snapshot}

INSTRUCTIONS:
1. Infer the user's "Identity Core" (Name, Occupation, Core Interests).
2. Determine their "Communication Style" based on how they speak (Concise, Detailed, Casual, Analytical).
3. Estimate "Preference Weights" (0.0 to 1.0).

OUTPUT STRICT JSON ONLY (Follow this schema):
{{
  "identity_core": {{
    "name": "string or null",
    "occupation": "string or null",
    "primary_interests": ["string", "string"],
    "communication_style": "concise | detailed | analytical | casual"
  }},
  "preference_weights": {{
    "tone_preference": <float 0.0=Formal, 1.0=Casual>,
    "detail_level": <float 0.0=Brief, 1.0=Detailed>,
    "risk_tolerance": <float 0.0=Safe, 1.0=Experimental>
  }},
  "behavioral_patterns": {{
    "common_topics": ["topic1", "topic2"]
  }},
  "confidence_score": <float 0.0-1.0>
}}

NOTE: Do not hallucinate. If a field is unknown, use null.
"""

class ProfileSynthesizer:
    def __init__(self, llm_client: LLMClientV2):
        self.llm = llm_client

    def synthesize(self, hybrid_memory) -> Dict[str, Any]:
        """
        Pull recent memories and preferences, then ask LLM to synthesize a profile update.
        """
        # 1. Gather Context (Snapshot)
        # We grab all preferences and recent conversations/episodes
        prefs = hybrid_memory.recall_preferences("identity interests job", top_k=20)
        episodes = hybrid_memory.recall_episodes("life event work project", top_k=10)
        recent_chat = hybrid_memory.long_term.get_conversation_history(limit=10)

        snapshot_text = self._format_snapshot(prefs, episodes, recent_chat)
        
        if not snapshot_text:
            logger.info("Not enough memory to synthesize profile.")
            return {}

        # 2. Prompt LLM
        prompt = SYNTHESIS_PROMPT.format(memory_snapshot=snapshot_text)
        
        try:
            # We explicitly do NOT query memory here to avoid circular context injection
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                query_for_memory=None 
            )
            
            return self._parse_json(response)
        except Exception as e:
            logger.error(f"Profile synthesis failed: {e}")
            return {}

    def _format_snapshot(self, prefs: List[dict], episodes: List[dict], chat: List[dict]) -> str:
        lines = ["--- PREFERENCES ---"]
        for p in prefs:
            lines.append(f"{p['key']}: {p['value']}")
            
        lines.append("\n--- RECENT EPISODES ---")
        for e in episodes:
            lines.append(f"{e.get('timestamp','')} - {e['event']}")
            
        lines.append("\n--- RECENT CONVERSATION STYLE ---")
        for c in chat:
            lines.append(f"User: {c['user_input']}")
            # We omit assistant response to focus on User's style
            
        return "\n".join(lines)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Attempt to extract JSON from LLM response."""
        import re
        try:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except:
            logger.warning("Failed to parse JSON from synthesis response")
            return {}
