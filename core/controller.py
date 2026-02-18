"""
core/controller.py
──────────────────
Jarvis Controller V5 (Session 7).
Implements Self-Model, Adaptive Profiling, and Session Synthesis.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

from core.llm.llm__init__ import LLMClientV2
from memory.hybrid_memory import HybridMemory
from memory.short_term import ShortTermMemory
from core.intelligence import MemoryIntelligence 
from core.intents import IntentClassifierV2, Intent
from core.safety import CommandSafetyGate

# Session 7 Imports
from core.profile import UserProfileEngine
from core.synthesis import ProfileSynthesizer

logger = logging.getLogger(__name__)

SESSION_OUTPUT_DIR = Path("outputs")

class JarvisControllerV5:
    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "D:/AI/Jarvis/data/chroma",
        model_name: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.session_id = f"Jarvis-Session-7-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # 1. Base Layers
        self.hybrid_memory = HybridMemory(db_path, chroma_path, embedding_model)
        self.short_term = ShortTermMemory()
        self.llm = LLMClientV2(self.hybrid_memory, model_name)
        
        # 2. Intelligence Layers
        self.memory_intelligence = MemoryIntelligence(self.llm)
        self.intent_classifier = IntentClassifierV2(self.llm)
        self.safety_gate = CommandSafetyGate()
        
        # 3. Identity & Profile Layer (Session 7)
        self.profile_engine = UserProfileEngine()
        self.synthesizer = ProfileSynthesizer(self.llm)

        # 4. Session State
        self.intent_stats = Counter()
        self._pending_clarification = None
        self._new_memories_count = 0  # Trigger synthesis after X updates

    def initialize(self) -> dict:
        logger.info(f"Initializing Jarvis V5 ({self.session_id})")
        
        # Ensure directories
        (SESSION_OUTPUT_DIR / self.session_id).mkdir(parents=True, exist_ok=True)
        
        mem_status = self.hybrid_memory.initialize()
        
        # Log profile status
        profile_summary = self.profile_engine.get_profile_summary()
        logger.info(f"Loaded User Profile:\n{profile_summary}")

        return {
            "session_id": self.session_id,
            "memory_mode": mem_status.get("mode", "unknown"),
            "profile_loaded": True
        }

    def process(self, user_input: str, stream: bool = False):
        if not user_input.strip(): return ""
        
        # 0. Handle Pending Clarification
        if self._pending_clarification:
            return self._resolve_clarification(user_input)

        # 1. Intent Classification
        classification = self.intent_classifier.classify(user_input)
        intent_str = classification.get("intent", "unknown").lower()
        confidence = classification.get("confidence", 0.0)
        
        self.intent_stats[intent_str] += 1
        self.profile_engine.log_interaction(intent_str)  # Update behavioral stats

        # 2. Adaptive Confidence Gate
        # Session 7: Lower threshold if we have high profile confidence
        base_threshold = 0.6
        if self.profile_engine.get_confidence_score() > 0.8:
            base_threshold = 0.4 # We know the user better, trust the AI more
            
        if confidence < base_threshold:
            self._pending_clarification = {"original_input": user_input, "guess": intent_str}
            return f"I'm not sure. Did you mean to {intent_str}? (yes/no)"

        # 3. Routing
        if intent_str == Intent.COMMAND.value:
            return self._handle_command_flow(user_input)
        elif intent_str == Intent.CHAT.value:
            return self._handle_chat(user_input, stream)
        elif intent_str == Intent.STORE_MEMORY.value:
            return self._handle_memory_store(user_input)
        elif intent_str == Intent.QUERY_MEMORY.value:
            return self._handle_memory_query(user_input)
        elif intent_str == Intent.META.value:
            return f"I am Jarvis (Session 7). Identity-Aware."
        else:
            return "I couldn't process that request."

    def _handle_chat(self, user_input: str, stream: bool = False):
        # Passive memory evaluation
        decision = self.memory_intelligence.evaluate_input(user_input)
        if decision and decision.get("decision") == "store":
            self._handle_implicit_storage(decision)

        # Get Profile Context for Adaptation
        profile_ctx = self.profile_engine.get_profile_summary()
        
        messages = [{"role": "user", "content": user_input}]
        
        if stream:
            return self.llm.chat_stream(messages, query_for_memory=user_input, profile_summary=profile_ctx)
        return self.llm.chat(messages, query_for_memory=user_input, profile_summary=profile_ctx)

    def _handle_memory_store(self, user_input: str) -> str:
        # Pass to intelligence logic
        decision = self.memory_intelligence.evaluate_input(user_input)
        if decision and decision.get("decision") == "store":
            return self._handle_implicit_storage(decision)
        return "I processed that, but didn't find a permanent fact to store."

    def _handle_implicit_storage(self, decision: dict) -> str:
        """Helper to store memory and track update count."""
        key = decision.get("key")
        val = decision.get("value")
        cat = decision.get("category")
        
        if cat == "preference":
            self.hybrid_memory.store_preference(key, val)
        elif cat == "episode":
            self.hybrid_memory.store_episode(val, category="observed")
            
        self._new_memories_count += 1
        
        # Session 7: Trigger Synthesis if we learned enough new things
        if self._new_memories_count >= 10:
            logger.info("Triggering mid-session profile synthesis...")
            self.run_synthesis()
            self._new_memories_count = 0
            
        return f"✓ stored: {key}={val}"

    def _handle_memory_query(self, user_input: str) -> str:
        context = self.hybrid_memory.build_context_block(user_input)
        if not context:
            return "I don't recall anything about that."
        return f"Here is what I found:\n{context}"

    def _handle_command_flow(self, user_input: str) -> str:
        safety = self.safety_gate.verify_command(user_input)
        if not safety["allowed"]:
            return f"🚫 {safety['reason']}"
            
        cmd = user_input.lower().split()[0]
        if cmd in ["exit", "quit"]:
            self.shutdown()
            return "__EXIT__"
        elif cmd == "status":
            return self._get_status_report()
        elif cmd == "synthesize":
            self.run_synthesis()
            return "✓ Manual profile synthesis complete."
            
        return f"✅ Executed: {cmd}"

    def run_synthesis(self):
        """Run the ProfileSynthesizer and update the UserProfile."""
        print("\n• Synthesizing Identity Profile...", end="", flush=True)
        delta = self.synthesizer.synthesize(self.hybrid_memory)
        if delta:
            self.profile_engine.update_profile(delta)
            print(" Done.")
            logger.info(f"Profile updated with delta: {delta.keys()}")
        else:
            print(" No changes detected.")

    def _get_status_report(self) -> str:
        p = self.profile_engine.profile
        return (
            f"✅ SYSTEM STATUS (Session 7):\n"
            f"Session ID: {self.session_id}\n"
            f"Confidence Score: {p.get('confidence_score', 0.0):.2f}\n"
            f"Identity: {p['identity_core'].get('name', 'Unknown')}\n"
            f"Style: {p['identity_core'].get('communication_style', 'Default')}\n"
            f"New Memories: {self._new_memories_count}"
        )

    def _resolve_clarification(self, user_input: str) -> str:
        # (Same logic as Session 6)
        original = self._pending_clarification["original_input"]
        guess = self._pending_clarification["guess"]
        self._pending_clarification = None
        
        if user_input.lower().startswith("y"):
            if guess == Intent.CHAT.value: return self._handle_chat(original)
            if guess == Intent.COMMAND.value: return self._handle_command_flow(original)
            return f"Confirmed {guess}."
        else:
            return "Understood. Please rephrase."

    def shutdown(self):
        """Run final synthesis and save artifacts."""
        print("\n• Shutting down Jarvis V5...")
        
        # Session 7: Final Synthesis
        self.run_synthesis()
        
        # Export Session Artifacts
        s_dir = SESSION_OUTPUT_DIR / self.session_id
        
        with open(s_dir / "user_profile_snapshot.json", "w") as f:
            json.dump(self.profile_engine.profile, f, indent=2)
            
        print(f"• Profile saved to memory/user_profile.json")
        print(f"• Session snapshot saved to {s_dir}")
