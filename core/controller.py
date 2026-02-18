"""
core/controller.py
──────────────────
Jarvis Controller V3 (Session 5).
Integrates Memory Intelligence and Self-Reflection.
"""

import logging
import uuid
import os
import json
from datetime import datetime
from pathlib import Path

from memory.hybrid_memory import HybridMemory
from memory.short_term import ShortTermMemory
from core.intents import classify_intent, Intent
from core.llm import LLMClientV2
from core.intelligence import MemoryIntelligence

logger = logging.getLogger(__name__)

SESSION_OUTPUT_DIR = Path("outputs")

class JarvisControllerV3:
    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "D:/AI/Jarvis/data/chroma",
        model_name: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.session_id   = f"Jarvis-Session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.started_at   = datetime.now()
        
        # 1. Memory Layers
        self.hybrid_memory = HybridMemory(db_path, chroma_path, embedding_model)
        self.short_term    = ShortTermMemory()
        
        # 2. Cognitive Layers
        self.llm          = LLMClientV2(self.hybrid_memory, model_name)
        self.intelligence = MemoryIntelligence(self.llm)

        self._initialized = False

    def initialize(self) -> dict:
        logger.info(f"Initializing Jarvis V3 ({self.session_id})")
        mem_status = self.hybrid_memory.initialize()
        ollama_ok  = self.llm.is_available()
        
        # Ensure output directory exists
        SESSION_OUTPUT_DIR.mkdir(exist_ok=True)
        (SESSION_OUTPUT_DIR / self.session_id).mkdir(exist_ok=True)

        self._initialized = True
        return {
            "session_id": self.session_id,
            "memory_mode": mem_status.get("mode"),
            "ollama": ollama_ok
        }

    def process(self, user_input: str, stream: bool = False) -> str:
        if not user_input.strip(): return ""

        # 1. Intent Classification
        intent = classify_intent(user_input)

        # 2. Intelligent Memory Check (Passive)
        # Even if not explicitly asked, check if this is a fact worth saving
        if intent in [Intent.MEMORY_STORE, Intent.UNKNOWN]:
            self._evaluate_and_store(user_input)

        # 3. Routing
        response = ""
        if intent == Intent.COMMAND:
            response = self._handle_command(user_input)
        elif intent == Intent.MEMORY_RECALL:
            # For recall, we query memory explicitly
            response = self.hybrid_memory.build_context_block(user_input)
            if not response:
                response = "I searched my memory but couldn't find a direct answer. Let me think..."
                # Fallback to LLM with empty context if needed
                response = self._handle_llm(user_input, stream)
        else:
            response = self._handle_llm(user_input, stream)

        # 4. Update Short Term Memory
        self.short_term.add_exchange(user_input, response if isinstance(response, str) else "...")
        return response

    def _evaluate_and_store(self, user_input: str):
        """Runs the Decision Gate."""
        logger.info("Running Memory Decision Gate...")
        decision = self.intelligence.evaluate_input(user_input)
        
        if decision and decision.get("decision") == "store":
            key = decision.get("key")
            val = decision.get("value")
            cat = decision.get("category")
            
            if cat == "preference" and key and val:
                self.hybrid_memory.store_preference(key, val)
                logger.info(f"🧠 Learned Preference: {key}={val}")
            elif cat == "episode":
                self.hybrid_memory.store_episode(val, category="observed")
                logger.info(f"🧠 Recorded Episode: {val}")

    def _handle_llm(self, user_input: str, stream: bool = False):
        messages = [{"role": "user", "content": user_input}]
        if stream:
            # In V3 we assume the caller handles the generator
            return self.llm.chat_stream(messages, query_for_memory=user_input)
        return self.llm.chat(messages, query_for_memory=user_input)

    def _handle_command(self, user_input: str) -> str:
        cmd = user_input.lower().strip()
        if cmd in ("exit", "quit", "bye"):
            self.shutdown()
            return "__EXIT__"
        if cmd == "status":
            return str(self.hybrid_memory.stats())
        return "Command not recognized."

    def shutdown(self):
        """End of session: Reflection and Artifact Saving."""
        print("\n• Initiating Session Shutdown Protocol...")
        
        # 1. Self-Reflection
        print("• Running Self-Reflection Module...")
        history = self.short_term.get_recent_context(num_turns=99)
        reflection = self.intelligence.perform_reflection(history)
        
        # 2. Save Artifacts
        session_dir = SESSION_OUTPUT_DIR / self.session_id
        
        with open(session_dir / "session_summary.json", "w") as f:
            json.dump(reflection, f, indent=2)
            
        with open(session_dir / "memory_stats.json", "w") as f:
            json.dump(self.hybrid_memory.stats(), f, indent=2)

        print(f"• Session artifacts saved to: {session_dir}")
        print("• Memory consolidated.")