"""
core/intents.py
════════════════
Intent classification for Jarvis V5.
Uses DeepSeek R1:8b to classify user input into structured intents.
Falls back to keyword heuristics if LLM is unavailable.
"""

import json
import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(Enum):
    CHAT          = "chat"
    COMMAND       = "command"
    STORE_MEMORY  = "store_memory"
    QUERY_MEMORY  = "query_memory"
    META          = "meta"
    UNKNOWN       = "unknown"


# ── Keyword fallback (no LLM needed) ──────────────────────
_COMMAND_KEYWORDS  = {"exit", "quit", "status", "synthesize", "reset", "help"}
_MEMORY_STORE_KW   = {"remember", "store", "save", "note that", "keep in mind"}
_MEMORY_QUERY_KW   = {"recall", "what do you know", "do you remember", "retrieve", "find memory"}
_META_KW           = {"who are you", "what are you", "jarvis", "your name", "version"}

CLASSIFY_SYSTEM = """You are an intent classifier for a personal AI assistant called Jarvis.

Classify the user message into EXACTLY ONE of these intents:
- chat          : general conversation, questions, requests for information
- command       : system commands (exit, quit, status, synthesize, help)
- store_memory  : user wants to save a fact, preference, or memory
- query_memory  : user wants to retrieve something Jarvis remembers
- meta          : questions about Jarvis itself

Respond ONLY with valid JSON, no explanation:
{"intent": "<intent>", "confidence": <0.0-1.0>}"""


class IntentClassifierV2:
    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, user_input: str) -> dict:
        """
        Returns {"intent": str, "confidence": float}
        Tries LLM first, falls back to keyword heuristic.
        """
        # Try LLM classification
        if self.llm is not None:
            result = self._llm_classify(user_input)
            if result:
                return result

        # Keyword fallback
        return self._keyword_classify(user_input)

    def _llm_classify(self, text: str) -> dict | None:
        """Synchronous LLM call for classification."""
        try:
            import asyncio
            prompt = f"Classify this message:\n\n{text}"
            # Run async in sync context
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self.llm.complete_json(prompt, system=CLASSIFY_SYSTEM, temperature=0.0)
            )
            loop.close()

            if result and "intent" in result:
                intent_str = result["intent"].lower()
                # Validate against known intents
                valid = {i.value for i in Intent}
                if intent_str not in valid:
                    intent_str = "unknown"
                return {
                    "intent": intent_str,
                    "confidence": float(result.get("confidence", 0.7))
                }
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
        return None

    def _keyword_classify(self, text: str) -> dict:
        """Fast keyword-based fallback classifier."""
        lower = text.lower().strip()
        first_word = lower.split()[0] if lower.split() else ""

        if first_word in _COMMAND_KEYWORDS:
            return {"intent": Intent.COMMAND.value, "confidence": 0.95}

        if any(kw in lower for kw in _MEMORY_STORE_KW):
            return {"intent": Intent.STORE_MEMORY.value, "confidence": 0.85}

        if any(kw in lower for kw in _MEMORY_QUERY_KW):
            return {"intent": Intent.QUERY_MEMORY.value, "confidence": 0.85}

        if any(kw in lower for kw in _META_KW):
            return {"intent": Intent.META.value, "confidence": 0.9}

        return {"intent": Intent.CHAT.value, "confidence": 0.75}
