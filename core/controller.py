"""
core/controller.py
──────────────────────
Updated Jarvis controller for Session 4.
"""

import logging
import uuid
import sys
from datetime import datetime
from typing import Optional

# FIX: Import correct module paths
from memory.hybrid_memory import HybridMemory
from memory.short_term import ShortTermMemory
from core.intents import classify_intent, Intent, extract_memory_data, extract_memory_query
# FIX: 'core.llm_v2' does not exist as a file, it is 'core.llm'
from core.llm import LLMClientV2

logger = logging.getLogger(__name__)

class JarvisControllerV2:
    """
    Central orchestration layer for Jarvis — Session 4 edition.
    """

    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "D:/AI/Jarvis/data/chroma",
        model_name: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.session_id   = str(uuid.uuid4())[:8]
        self.started_at   = datetime.now()
        self.exchange_count = 0

        # Memory layers
        self.hybrid_memory = HybridMemory(
            db_path=db_path,
            chroma_path=chroma_path,
            model_name=embedding_model,
        )
        self.short_term = ShortTermMemory()

        # LLM
        self.llm = LLMClientV2(
            hybrid_memory=self.hybrid_memory,
            model=model_name,
        )

        self._initialized = False

    def initialize(self) -> dict:
        logger.info(f"Initializing Jarvis (session: {self.session_id})")
        memory_status = self.hybrid_memory.initialize()
        ollama_ok     = self.llm.is_available()
        self._initialized = True
        return {
            "session_id":   self.session_id,
            "memory_mode":  memory_status.get("mode", "unknown"),
            "ollama":       ollama_ok,
        }

    def process(self, user_input: str, stream: bool = False) -> str:
        if not self._initialized:
            self.initialize()

        user_input = user_input.strip()
        if not user_input:
            return ""

        # Classify intent
        intent = classify_intent(user_input)
        
        # Route
        if intent == Intent.COMMAND:
            return self._handle_command(user_input)
        elif intent == Intent.MEMORY_STORE:
            return self._handle_memory_store(user_input)
        elif intent == Intent.MEMORY_RECALL:
            return self._handle_memory_recall(user_input)
        else:
            return self._handle_llm(user_input, stream=stream)

    def _handle_memory_store(self, user_input: str) -> str:
        pairs = extract_memory_data(user_input)
        if not pairs:
            return "I'd like to remember that, but I couldn't extract specific information."

        stored = []
        for key, value in pairs.items():
            self.hybrid_memory.store_preference(key, value)
            stored.append(f"{key} = {value}")

        self.hybrid_memory.store_episode(f"User said: {user_input}", category="preference_update")
        return f"✓ Got it — I'll remember: {', '.join(stored)}"

    def _handle_memory_recall(self, user_input: str) -> str:
        query = extract_memory_query(user_input) or user_input
        results = self.hybrid_memory.recall_all(query, top_k=5)
        
        # Simplified response logic for brevity
        if not any(results.values()):
            return "I don't recall anything about that."
            
        return self.hybrid_memory.build_context_block(query)

    def _handle_command(self, user_input: str) -> str:
        cmd = user_input.lower().strip()
        if cmd in ("exit", "quit", "bye"):
            return "__EXIT__"
        if cmd == "status":
            return str(self.hybrid_memory.stats())
        return self._handle_llm(user_input)

    def _handle_llm(self, user_input: str, stream: bool = False) -> str:
        messages = [{"role": "user", "content": user_input}]
        if stream:
            # Simple stream handler
            print(f"Jarvis: ", end="", flush=True)
            full_resp = ""
            for chunk in self.llm.chat_stream(messages, query_for_memory=user_input):
                print(chunk, end="", flush=True)
                full_resp += chunk
            print()
            return full_resp
        return self.llm.chat(messages, query_for_memory=user_input)