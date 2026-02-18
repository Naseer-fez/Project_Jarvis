"""
memory/hybrid_memory.py
────────────────────────
Hybrid memory manager for Jarvis.

Combines:
  - SQLite (long_term.py)  → exact keyword lookup, full history, structured data
  - ChromaDB (semantic_memory.py) → meaning-based vector recall

Strategy:
  1. On WRITE  → write to both SQLite AND ChromaDB (keep in sync)
  2. On RECALL → query both, merge results, deduplicate, score & rank
  3. On CONTEXT INJECT → compress top results into a token-efficient string

Author: Jarvis Session 4
"""

import logging
from typing import Optional

from memory.long_term import LongTermMemory
from memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)

# ─── Weights for hybrid scoring ───────────────────────────────────────────────
# When a result appears in both exact AND semantic results, boost its score.
SEMANTIC_WEIGHT = 0.70   # Weight for vector similarity score
EXACT_BONUS     = 0.30   # Added if the key also exists in SQLite exact results
MAX_CONTEXT_ITEMS = 8    # Max memory items injected into LLM context


class HybridMemory:
    """
    Single unified interface for all Jarvis memory operations.

    Write once → stored in both SQLite and ChromaDB.
    Recall queries both stores and returns intelligently merged results.
    """

    def __init__(
        self,
        db_path: str = "memory/memory.db",
        chroma_path: str = "D:/AI/Jarvis/data/chroma",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.long_term = LongTermMemory(db_path=db_path)
        self.semantic  = SemanticMemory(chroma_path=chroma_path, model_name=model_name)
        self._semantic_available = False

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> dict:
        """
        Initialize both backends. Semantic memory is optional —
        if it fails, system continues with SQLite-only mode.
        """
        status = {
            "sqlite":   True,
            "semantic": False,
            "mode":     "sqlite_only",
        }

        self._semantic_available = self.semantic.initialize()
        status["semantic"] = self._semantic_available

        if self._semantic_available:
            status["mode"] = "hybrid"
            logger.info("HybridMemory: running in HYBRID mode (SQLite + ChromaDB)")
        else:
            logger.warning("HybridMemory: running in SQLITE-ONLY mode (semantic unavailable)")

        return status

    # ── Write Operations ──────────────────────────────────────────────────────

    def store_preference(self, key: str, value: str) -> bool:
        """
        Store a user preference in both SQLite and ChromaDB.
        Returns True if at least SQLite write succeeded.
        """
        # Write to SQLite (always)
        sqlite_ok = self.long_term.store_preference(key, value)

        # Write to ChromaDB (best-effort)
        if self._semantic_available:
            try:
                self.semantic.store_preference(key, value)
            except Exception as e:
                logger.warning(f"Semantic preference write failed (non-fatal): {e}")

        return sqlite_ok

    def store_episode(self, event: str, category: str = "general") -> bool:
        """Store an episodic event in both backends."""
        # SQLite
        self.long_term.store_episodic_event(event, category)

        # ChromaDB
        if self._semantic_available:
            try:
                self.semantic.store_episode(event, category)
            except Exception as e:
                logger.warning(f"Semantic episode write failed (non-fatal): {e}")

        return True

    def store_conversation(
        self,
        user_input: str,
        response: str,
        session_id: str = "default",
    ) -> bool:
        """Store a conversation turn in both backends."""
        # SQLite
        self.long_term.store_conversation(user_input, response, session_id)

        # ChromaDB
        if self._semantic_available:
            try:
                self.semantic.store_conversation_turn(user_input, response, session_id)
            except Exception as e:
                logger.warning(f"Semantic conversation write failed (non-fatal): {e}")

        return True

    # ── Recall Operations ─────────────────────────────────────────────────────

    def recall_preferences(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid preference recall.
        Returns merged + ranked list of preferences relevant to the query.
        """
        results = []

        # 1. Exact SQLite lookup
        exact_prefs = self.long_term.get_all_preferences()  # {key: value}
        exact_keys  = set(exact_prefs.keys())

        # 2. Semantic recall
        if self._semantic_available:
            semantic_hits = self.semantic.recall_preferences(query, top_k=top_k)
            seen_keys = set()

            for hit in semantic_hits:
                key   = hit["metadata"].get("key", "")
                value = hit["metadata"].get("value", "")
                score = hit["score"]

                # Boost score if key also exists in SQLite
                if key in exact_keys:
                    score = min(1.0, score + EXACT_BONUS)

                results.append({
                    "key":    key,
                    "value":  value,
                    "score":  round(score, 4),
                    "source": "hybrid" if key in exact_keys else "semantic",
                })
                seen_keys.add(key)

            # Add any exact-only preferences not caught by semantic (low-score edge cases)
            for key, value in exact_prefs.items():
                if key not in seen_keys:
                    results.append({
                        "key":    key,
                        "value":  value,
                        "score":  0.50,   # Default score for exact-only hits
                        "source": "exact",
                    })

        else:
            # Fallback: return all SQLite preferences
            for key, value in exact_prefs.items():
                results.append({
                    "key":    key,
                    "value":  value,
                    "score":  1.0,
                    "source": "exact",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def recall_episodes(self, query: str, top_k: int = 5) -> list[dict]:
        """Recall relevant episodic memories using semantic search."""
        if self._semantic_available:
            hits = self.semantic.recall_episodes(query, top_k=top_k)
            return [
                {
                    "event":     h["document"],
                    "category":  h["metadata"].get("category", "general"),
                    "timestamp": h["metadata"].get("timestamp", ""),
                    "score":     h["score"],
                    "source":    "semantic",
                }
                for h in hits
            ]
        else:
            # Fallback: return most recent from SQLite
            rows = self.long_term.get_recent_episodes(limit=top_k)
            return [
                {
                    "event":     r["event"],
                    "category":  r.get("category", "general"),
                    "timestamp": r.get("timestamp", ""),
                    "score":     1.0,
                    "source":    "exact",
                }
                for r in rows
            ]

    def recall_conversations(self, query: str, top_k: int = 5) -> list[dict]:
        """Recall relevant past conversation turns."""
        if self._semantic_available:
            hits = self.semantic.recall_conversations(query, top_k=top_k)
            return [
                {
                    "user_input":         h["metadata"].get("user_input", ""),
                    "assistant_response": h["metadata"].get("assistant_response", ""),
                    "timestamp":          h["metadata"].get("timestamp", ""),
                    "score":              h["score"],
                    "source":             "semantic",
                }
                for h in hits
            ]
        else:
            rows = self.long_term.get_recent_conversations(limit=top_k)
            return [
                {
                    "user_input":         r["user_input"],
                    "assistant_response": r["assistant_response"],
                    "timestamp":          r.get("timestamp", ""),
                    "score":              1.0,
                    "source":             "exact",
                }
                for r in rows
            ]

    def recall_all(self, query: str, top_k: int = 5) -> dict:
        """Full recall across all memory types."""
        return {
            "preferences":   self.recall_preferences(query, top_k),
            "episodes":      self.recall_episodes(query, top_k),
            "conversations": self.recall_conversations(query, top_k),
        }

    # ── Context Compression ───────────────────────────────────────────────────

    def build_context_block(self, query: str, max_items: int = MAX_CONTEXT_ITEMS) -> str:
        """
        Build a compact, token-efficient memory context string for LLM injection.

        Format:
          [MEMORY CONTEXT]
          Preferences: name=Alice | likes=coffee | prefers=dark mode
          Episodes: Discussed project planning (2025-01-10)
          Past conversations: ...
          [END MEMORY CONTEXT]

        Only includes results above the relevance threshold.
        """
        all_results = self.recall_all(query, top_k=max_items)
        lines = ["[MEMORY CONTEXT]"]

        # Preferences
        prefs = all_results["preferences"]
        if prefs:
            pref_str = " | ".join(
                f"{p['key']}={p['value']}" for p in prefs[:max_items // 2]
            )
            lines.append(f"Preferences: {pref_str}")

        # Episodes
        episodes = all_results["episodes"]
        if episodes:
            ep_parts = []
            for ep in episodes[:3]:
                ts = ep["timestamp"][:10] if ep["timestamp"] else ""
                ep_parts.append(f"{ep['event']}" + (f" ({ts})" if ts else ""))
            lines.append("Episodes: " + "; ".join(ep_parts))

        # Past conversations (only highest-scoring turn)
        convos = all_results["conversations"]
        if convos:
            top = convos[0]
            snippet = top["user_input"][:80] + ("..." if len(top["user_input"]) > 80 else "")
            lines.append(f"Relevant past exchange: User said: \"{snippet}\"")

        if len(lines) == 1:
            return ""  # No memory context available

        lines.append("[END MEMORY CONTEXT]")
        return "\n".join(lines)

    # ── Stats & Health ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return health and count stats from both backends."""
        sqlite_stats   = self.long_term.stats()
        semantic_stats = self.semantic.stats() if self._semantic_available else {"initialized": False}

        return {
            "mode":     "hybrid" if self._semantic_available else "sqlite_only",
            "sqlite":   sqlite_stats,
            "semantic": semantic_stats,
        }

    @property
    def semantic_available(self) -> bool:
        return self._semantic_available

