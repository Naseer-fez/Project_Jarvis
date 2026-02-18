"""
core/controller_v2.py
──────────────────────
Updated Jarvis controller for Session 4.

Changes from Session 3 (core/controller.py):
  - Uses HybridMemory (SQLite + ChromaDB) instead of LongTermMemory directly
  - LLMClientV2 with semantic context injection
  - Semantic recall path for MEMORY_RECALL intents
  - Conversation turns written to both SQLite and ChromaDB
  - Session summary includes semantic memory stats

Author: Jarvis Session 4
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from memory.hybrid_memory import HybridMemory
from memory.short_term import ShortTermMemory
from core.intents import classify_intent, Intent, extract_memory_data, extract_memory_query
from core.llm_v2 import LLMClientV2

logger = logging.getLogger(__name__)


class JarvisControllerV2:
    """
    Central orchestration layer for Jarvis — Session 4 edition.

    Flow per user input:
      1. Classify intent
      2. Route to handler (memory store / recall / question / command)
      3. Update short-term + hybrid memory
      4. Return response string
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

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> dict:
        """
        Initialize all subsystems. Returns a status dict.
        """
        logger.info(f"Initializing Jarvis (session: {self.session_id})")

        memory_status = self.hybrid_memory.initialize()
        ollama_ok     = self.llm.is_available()

        self._initialized = True

        status = {
            "session_id":   self.session_id,
            "memory_mode":  memory_status.get("mode", "unknown"),
            "sqlite":       memory_status.get("sqlite", False),
            "semantic":     memory_status.get("semantic", False),
            "ollama":       ollama_ok,
        }

        logger.info(f"Startup status: {status}")
        return status

    # ── Main Input Handler ────────────────────────────────────────────────────

    def process(self, user_input: str, stream: bool = False) -> str:
        """
        Process a user input and return Jarvis's response.

        Args:
            user_input: Raw text from the user.
            stream:     If True, prints streamed output and returns full string.
        """
        if not self._initialized:
            self.initialize()

        user_input = user_input.strip()
        if not user_input:
            return ""

        # Classify intent
        intent = classify_intent(user_input)
        logger.debug(f"[{self.session_id}] Intent: {intent} | Input: {user_input[:60]}")

        # Route
        if intent == Intent.COMMAND:
            response = self._handle_command(user_input)

        elif intent == Intent.MEMORY_STORE:
            response = self._handle_memory_store(user_input)

        elif intent == Intent.MEMORY_RECALL:
            response = self._handle_memory_recall(user_input)

        else:
            # QUESTION or UNKNOWN → send to LLM
            response = self._handle_llm(user_input, stream=stream)

        # Update short-term memory buffer
        self.short_term.add_turn(user_input, response)
        self.exchange_count += 1

        # Write to hybrid memory (async-safe, non-blocking errors)
        try:
            self.hybrid_memory.store_conversation(user_input, response, self.session_id)
        except Exception as e:
            logger.warning(f"Failed to store conversation turn: {e}")

        return response

    # ── Intent Handlers ───────────────────────────────────────────────────────

    def _handle_memory_store(self, user_input: str) -> str:
        """Extract and store a preference/fact from the user's input."""
        pairs = extract_memory_data(user_input)

        if not pairs:
            return "I'd like to remember that, but I couldn't extract specific information. Could you rephrase it?"

        stored = []
        for key, value in pairs.items():
            self.hybrid_memory.store_preference(key, value)
            stored.append(f"{key} = {value}")

        # Also log as episodic event
        event = f"User said: {user_input}"
        self.hybrid_memory.store_episode(event, category="preference_update")

        if len(stored) == 1:
            return f"✓ Got it — I'll remember: {stored[0]}"
        return f"✓ Noted! I'll remember:\n" + "\n".join(f"  • {s}" for s in stored)

    def _handle_memory_recall(self, user_input: str) -> str:
        """
        Perform semantic recall for the user's query.
        Falls back to keyword match if semantic isn't available.
        """
        query = extract_memory_query(user_input) or user_input
        results = self.hybrid_memory.recall_all(query, top_k=5)

        prefs  = results.get("preferences", [])
        eps    = results.get("episodes", [])
        convos = results.get("conversations", [])

        if not prefs and not eps and not convos:
            return "I don't have any relevant memories for that yet."

        lines = ["Here's what I remember:"]

        if prefs:
            lines.append("\n📌 About you:")
            for p in prefs:
                score_note = f" (relevance: {p['score']:.0%})" if p["score"] < 0.8 else ""
                lines.append(f"  • {p['key']}: {p['value']}{score_note}")

        if eps:
            lines.append("\n📖 Past events:")
            for ep in eps[:3]:
                lines.append(f"  • {ep['event']}")

        if convos and not prefs:
            lines.append("\n💬 From our conversations:")
            for c in convos[:2]:
                snippet = c["user_input"][:80]
                lines.append(f"  • You once said: \"{snippet}\"")

        return "\n".join(lines)

    def _handle_command(self, user_input: str) -> str:
        """Handle built-in commands."""
        cmd = user_input.lower().strip()

        if cmd in ("exit", "quit", "bye"):
            return "__EXIT__"

        if cmd == "help":
            return (
                "Available commands:\n"
                "  status        — Show session info and memory stats\n"
                "  memory        — Show all stored preferences\n"
                "  recall <text> — Semantic search across all memories\n"
                "  clear memory  — Wipe all stored preferences\n"
                "  exit / quit   — End session\n\n"
                "Or just talk naturally — I'll understand what you mean."
            )

        if cmd == "status":
            return self._status_report()

        if cmd == "memory" or cmd == "show memory":
            return self._show_all_memory()

        if cmd.startswith("recall "):
            query = cmd[7:].strip()
            return self._handle_memory_recall(query)

        if cmd == "clear memory":
            return "⚠️  Memory clear not implemented in this session (safety guard). Use the database directly."

        # Unknown command → pass to LLM
        return self._handle_llm(user_input)

    def _handle_llm(self, user_input: str, stream: bool = False) -> str:
        """Send input to the LLM with conversation history + semantic context."""
        # Build conversation history from short-term memory
        history  = self.short_term.get_formatted_context()
        messages = []

        # Inject short-term conversation turns
        for turn in self.short_term.get_turns():
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant",  "content": turn["assistant"]})

        # Add current user message
        messages.append({"role": "user", "content": user_input})

        if stream:
            response_parts = []
            for chunk in self.llm.chat_stream(messages, query_for_memory=user_input):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
            print()  # newline after streaming
            return "".join(response_parts)
        else:
            return self.llm.chat(messages, query_for_memory=user_input)

    # ── Reports ───────────────────────────────────────────────────────────────

    def _status_report(self) -> str:
        """Build a session status string."""
        stats       = self.hybrid_memory.stats()
        sem_stats   = stats.get("semantic", {})
        sqlite_stats = stats.get("sqlite", {})
        uptime      = datetime.now() - self.started_at
        minutes     = int(uptime.total_seconds() // 60)
        seconds     = int(uptime.total_seconds() % 60)

        semantic_line = (
            f"  Semantic memory : {sem_stats.get('preferences', 0)} prefs | "
            f"{sem_stats.get('episodes', 0)} episodes | "
            f"{sem_stats.get('conversations', 0)} conversations"
            if sem_stats.get("initialized")
            else "  Semantic memory : unavailable (SQLite-only mode)"
        )

        return (
            f"📊 Jarvis Session Status\n"
            f"  Session ID      : {self.session_id}\n"
            f"  Uptime          : {minutes}m {seconds}s\n"
            f"  Exchanges       : {self.exchange_count}\n"
            f"  Memory mode     : {stats.get('mode', 'unknown')}\n"
            f"  SQLite prefs    : {sqlite_stats.get('preference_count', '?')}\n"
            f"{semantic_line}\n"
            f"  LLM             : {'✓ Online' if self.llm.is_available() else '✗ Offline'}\n"
            f"  Model           : {self.llm.model}"
        )

    def _show_all_memory(self) -> str:
        """Fetch and display all stored preferences."""
        prefs = self.hybrid_memory.long_term.get_all_preferences()
        if not prefs:
            return "No preferences stored yet."
        lines = ["📋 Stored preferences:"]
        for key, value in prefs.items():
            lines.append(f"  • {key}: {value}")
        return "\n".join(lines)

    def session_summary(self) -> dict:
        """Return a dict summarizing the session — called on shutdown."""
        stats = self.hybrid_memory.stats()
        return {
            "session_id":    self.session_id,
            "exchanges":     self.exchange_count,
            "memory_mode":   stats.get("mode"),
            "semantic_stats": stats.get("semantic", {}),
            "sqlite_stats":  stats.get("sqlite", {}),
        }
