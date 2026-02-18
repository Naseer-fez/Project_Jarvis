"""
core/controller.py
──────────────────
Jarvis Controller V4 (Session 6).
Implements Routing Discipline, Confidence Gates, and Safety Checks.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

from core.llm import LLMClientV2
from memory.hybrid_memory import HybridMemory
from memory.short_term import ShortTermMemory
# Import Session 5 Intelligence
from core.intelligence import MemoryIntelligence 
# Import Session 6 Components
from core.intents import IntentClassifierV2, Intent
from core.safety import CommandSafetyGate

logger = logging.getLogger(__name__)

SESSION_OUTPUT_DIR = Path("outputs")

class JarvisControllerV4:
    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "D:/AI/Jarvis/data/chroma",
        model_name: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.session_id = f"Jarvis-Session-6-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Core Components
        self.hybrid_memory = HybridMemory(db_path, chroma_path, embedding_model)
        self.short_term = ShortTermMemory()
        self.llm = LLMClientV2(self.hybrid_memory, model_name)
        
        # Intelligence Layers
        self.memory_intelligence = MemoryIntelligence(self.llm)
        self.intent_classifier = IntentClassifierV2(self.llm)
        self.safety_gate = CommandSafetyGate()

        # Session Artifacts
        self.intent_stats = Counter()
        self.refused_commands = []
        self.clarification_events = []
        
        # State
        self._pending_clarification = None  # To handle multi-turn clarification
        self._initialized = False

    def initialize(self) -> dict:
        logger.info(f"Initializing Jarvis V4 ({self.session_id})")
        
        # Ensure output directories
        (SESSION_OUTPUT_DIR / self.session_id).mkdir(parents=True, exist_ok=True)
        
        mem_status = self.hybrid_memory.initialize()
        self._initialized = True
        
        return {
            "session_id": self.session_id,
            "system_status": "SECURE",
            "safety_gate": "ACTIVE"
        }

    def process(self, user_input: str, stream: bool = False) -> str:
        if not user_input.strip(): return ""
        
        # 0. Handle Pending Clarification (Multi-turn logic)
        if self._pending_clarification:
            return self._resolve_clarification(user_input)

        # 1. Intent Classification
        classification = self.intent_classifier.classify(user_input)
        intent_str = classification.get("intent", "unknown").lower()
        confidence = classification.get("confidence", 0.0)
        
        self.intent_stats[intent_str] += 1
        logger.info(f"Input: '{user_input}' | Intent: {intent_str} ({confidence})")

        # 2. Confidence Gate (Min 0.6)
        if confidence < 0.6:
            self.clarification_events.append({
                "timestamp": str(datetime.now()),
                "input": user_input,
                "classified_as": intent_str,
                "confidence": confidence
            })
            self._pending_clarification = {"original_input": user_input, "guess": intent_str}
            return f"I'm not 100% sure what you mean. Did you mean to {intent_str}? (yes/no/correction)"

        # 3. Routing Discipline
        if intent_str == Intent.COMMAND.value:
            return self._handle_command_flow(user_input)
            
        elif intent_str == Intent.STORE_MEMORY.value:
            return self._handle_memory_store(user_input)
            
        elif intent_str == Intent.QUERY_MEMORY.value:
            return self._handle_memory_query(user_input)
            
        elif intent_str == Intent.META.value:
            return self._handle_meta(user_input)
            
        elif intent_str == Intent.CHAT.value:
            return self._handle_chat(user_input, stream)
            
        else:
            return "I couldn't process that request safely."

    def _handle_command_flow(self, user_input: str) -> str:
        """Strict Command Safety Gate Flow"""
        safety_check = self.safety_gate.verify_command(user_input)
        
        if not safety_check["allowed"]:
            self.refused_commands.append({
                "timestamp": str(datetime.now()),
                "command": user_input,
                "reason": safety_check["reason"]
            })
            return f"🚫 COMMAND REFUSED: {safety_check['reason']}"

        # Dry Run / Execution Info
        dry_run = safety_check.get("dry_run", "Executing...")
        
        # Execute Safe Commands
        cmd = user_input.lower().split()[0]
        if cmd in ["exit", "quit", "bye"]:
            self.shutdown()
            return "__EXIT__"
        elif cmd == "status":
            return f"✅ SYSTEM STATUS:\nSession: {self.session_id}\nMemory: Online\nSafety: Active"
        elif cmd == "help":
            return "Available Commands: help, status, clear, exit, whoami"
        
        return f"✅ {dry_run}"

    def _handle_memory_store(self, user_input: str) -> str:
        """Route to Session 5 Memory Intelligence"""
        self.memory_intelligence.evaluate_input(user_input) # Logic handled inside Intelligence
        return "Thinking about whether to remember that..." # Placeholder, Intelligence logs it

    def _handle_memory_query(self, user_input: str) -> str:
        """Explicit Memory Recall"""
        context = self.hybrid_memory.build_context_block(user_input)
        if not context:
            return "I checked my memory banks, but found nothing specific on that."
        return f"Here is what I remember:\n{context}"

    def _handle_meta(self, user_input: str) -> str:
        """Meta-Awareness (No Memory Pollution)"""
        return f"I am Jarvis (Session 6). I am running locally with a Safety Gate active. My session ID is {self.session_id}."

    def _handle_chat(self, user_input: str, stream: bool = False):
        """Standard Chat Flow"""
        # Session 5 Integration: Check for passive memory storage even in chat
        self.memory_intelligence.evaluate_input(user_input)
        
        messages = [{"role": "user", "content": user_input}]
        if stream:
            return self.llm.chat_stream(messages, query_for_memory=user_input)
        return self.llm.chat(messages, query_for_memory=user_input)

    def _resolve_clarification(self, user_input: str) -> str:
        """Handle user response to low-confidence gate"""
        original = self._pending_clarification["original_input"]
        guess = self._pending_clarification["guess"]
        self._pending_clarification = None # Reset
        
        if user_input.lower().startswith("y"):
            # User confirmed the guess
            # Recursively call process with the original input, bypassing check (simulated)
            # In a real system, we'd inject the confirmed intent. 
            # For now, we just route it manually based on the confirmed guess.
            if guess == Intent.COMMAND.value: return self._handle_command_flow(original)
            if guess == Intent.CHAT.value: return self._handle_chat(original)
            return f"Okay, proceeding with {guess} logic for '{original}'."
        else:
            return "Understood. Please rephrase your request so I can understand better."

    def shutdown(self):
        """Save Session 6 Artifacts"""
        print("\n• Saving Session 6 Audit Logs...")
        s_dir = SESSION_OUTPUT_DIR / self.session_id
        
        with open(s_dir / "intent_stats.json", "w") as f:
            json.dump(dict(self.intent_stats), f, indent=2)
            
        with open(s_dir / "refused_commands.log", "w") as f:
            json.dump(self.refused_commands, f, indent=2)
            
        with open(s_dir / "clarification_events.log", "w") as f:
            json.dump(self.clarification_events, f, indent=2)
            
        print(f"• Audit complete. stored in {s_dir}")