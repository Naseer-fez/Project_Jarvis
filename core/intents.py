"""
core/intents.py
───────────────
Intent Classification V2 (Session 6).
Returns strict JSON with confidence scores and permission flags.
"""

import json
import re
import logging
from enum import Enum
from typing import Dict, Any, Optional
from core.llm import LLMClientV2

logger = logging.getLogger(__name__)

class Intent(Enum):
    CHAT = "chat"
    QUERY_MEMORY = "query_memory"
    STORE_MEMORY = "store_memory"
    COMMAND = "command"
    META = "meta"         # New: Questions about Jarvis itself
    UNKNOWN = "unknown"

INTENT_PROMPT = """
You are the Intent Classifier for Jarvis.
Analyze the USER INPUT and categorize it into exactly one category.

CATEGORIES:
1. COMMAND: User wants to change system state (exit, clear, help, status).
2. QUERY_MEMORY: User asks about past facts, preferences, or saved data ("What did I say about...", "Do you know...").
3. STORE_MEMORY: User explicitly wants to save info ("Remember that...", "I live in...").
4. META: Questions about YOU (the AI), your version, or this session.
5. CHAT: General conversation, greetings, jokes, thoughts.

USER INPUT: "{user_input}"

OUTPUT STRICT JSON ONLY:
{{
  "intent": "COMMAND" | "QUERY_MEMORY" | "STORE_MEMORY" | "META" | "CHAT" | "UNKNOWN",
  "confidence": <float 0.0-1.0>,
  "allowed": <boolean>
}}
"""

class IntentClassifierV2:
    def __init__(self, llm_client: LLMClientV2):
        self.llm = llm_client

    def classify(self, user_input: str) -> Dict[str, Any]:
        """
        Hybrid Classification:
        1. Regex (Fast, 1.0 confidence)
        2. LLM (Slow, variable confidence)
        """
        clean_input = user_input.strip().lower()

        # 1. Fast Path: Regex for Hardcoded Commands
        if clean_input in ["exit", "quit", "bye", "help", "status", "clear", "whoami"]:
            return {"intent": Intent.COMMAND.value, "confidence": 1.0, "allowed": True}

        # 2. Slow Path: LLM Analysis
        try:
            prompt = INTENT_PROMPT.format(user_input=user_input)
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                query_for_memory=None # Do not pollute classifier with memory context
            )
            parsed = self._parse_json(response)
            
            if parsed and "intent" in parsed:
                return parsed
            
        except Exception as e:
            logger.error(f"Intent Classification failed: {e}")

        # Fallback
        return {"intent": Intent.UNKNOWN.value, "confidence": 0.0, "allowed": False}

    def _parse_json(self, text: str) -> Optional[Dict]:
        try:
            # Attempt to find JSON structure even if LLM chats around it
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except:
            return None