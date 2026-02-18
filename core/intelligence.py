"""
core/intelligence.py
────────────────────
Memory Intelligence Layer (Session 5).
Handles the "Decision Gate" and "Self-Reflection" logic.
"""

import json
import logging
import re
from typing import Optional, Dict, Any

from core.llm import LLMClientV2
from core.prompts import MEMORY_EVALUATION_PROMPT, REFLECTION_PROMPT

logger = logging.getLogger(__name__)

class MemoryIntelligence:
    """
    The cognitive layer that sits between User Input and Memory Storage.
    """

    def __init__(self, llm_client: LLMClientV2):
        self.llm = llm_client

    def evaluate_input(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Decision Gate: Asks LLM if input is worth remembering.
        Returns dict with keys {key, value, category} if yes, else None.
        """
        # Fast path: Ignore very short inputs
        if len(user_input.split()) < 3:
            return None

        prompt = MEMORY_EVALUATION_PROMPT.format(user_input=user_input)
        
        # We use a separate context-free call for evaluation to be objective
        # Passing 'query_for_memory=None' prevents injecting existing memory to avoid bias
        raw_response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            query_for_memory=None 
        )

        return self._parse_json(raw_response)

    def perform_reflection(self, session_history: list) -> Dict[str, Any]:
        """
        End-of-Session Reflection: Analyzes the conversation log.
        """
        if not session_history:
            return {"summary": "Empty session", "new_learned_facts": []}

        # Convert history objects to string log
        log_text = "\n".join([
            f"User: {turn['user']}\nJarvis: {turn['assistant']}"
            for turn in session_history
        ])

        prompt = REFLECTION_PROMPT.format(conversation_log=log_text)
        
        raw_response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            query_for_memory=None
        )

        result = self._parse_json(raw_response)
        if not result:
            return {"summary": "Reflection failed", "error": "Invalid JSON"}
        return result

    def _parse_json(self, text: str) -> Optional[Dict]:
        """
        Robust JSON extractor that handles LLM 'thinking' output or markdown code blocks.
        """
        try:
            # 1. Try direct parse
            return json.loads(text)
        except:
            pass

        # 2. Extract from markdown code blocks ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # 3. Extract first valid {...} block found in text
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        logger.warning(f"Failed to parse JSON from LLM response: {text[:50]}...")
        return None
