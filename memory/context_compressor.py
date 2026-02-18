"""
core/context_compressor.py
───────────────────────────
Context compression and memory injection for Jarvis.

Problem: LLMs have a context window limit. Naively injecting all memory
entries causes token bloat, distraction, and slow responses.

Solution: This module selects and formats only the most RELEVANT memory
entries for a given query, compressed into a minimal, structured block.

Strategies:
  1. Relevance filtering   — keep only items above a score threshold
  2. Deduplication         — remove redundant/near-identical entries
  3. Token estimation      — rough word-count budget to stay within limits
  4. Priority ordering     — preferences > episodes > conversations
  5. Truncation            — trim long strings to fit budget

Author: Jarvis Session 4
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS  = 400    # Rough token budget for memory context block
DEFAULT_THRESHOLD   = 0.30   # Minimum relevance score to include
MAX_PREF_ITEMS      = 6      # Max preference lines
MAX_EPISODE_ITEMS   = 3      # Max episode lines
MAX_CONVO_ITEMS     = 2      # Max conversation snippets
MAX_VALUE_LEN       = 60     # Max chars per preference value
MAX_EPISODE_LEN     = 100    # Max chars per episode
MAX_CONVO_LEN       = 120    # Max chars per conversation snippet


class ContextCompressor:
    """
    Compresses hybrid memory recall results into a compact LLM-ready string.

    Usage:
        cc = ContextCompressor()
        context_block = cc.compress(query, recall_results)
        # → inject into LLM system prompt
    """

    def __init__(
        self,
        max_tokens: int   = DEFAULT_MAX_TOKENS,
        threshold:  float = DEFAULT_THRESHOLD,
    ):
        self.max_tokens = max_tokens
        self.threshold  = threshold

    # ── Main Entry ────────────────────────────────────────────────────────────

    def compress(
        self,
        query: str,
        recall_results: dict,
        include_scores: bool = False,
    ) -> str:
        """
        Compress recall_results dict into a minimal context block string.

        Args:
            query:          The user's current input (used for relevance context).
            recall_results: Output of HybridMemory.recall_all() — dict with keys:
                            'preferences', 'episodes', 'conversations'
            include_scores: If True, append relevance score to each item (debug).

        Returns:
            A formatted string block, or empty string if nothing relevant found.
        """
        lines = []
        token_budget = self.max_tokens

        # 1. Preferences (highest priority)
        pref_lines, tokens_used = self._compress_preferences(
            recall_results.get("preferences", []),
            include_scores=include_scores,
        )
        if pref_lines:
            lines.append("Preferences: " + " | ".join(pref_lines))
            token_budget -= tokens_used

        # 2. Episodes
        if token_budget > 50:
            ep_lines, tokens_used = self._compress_episodes(
                recall_results.get("episodes", []),
                include_scores=include_scores,
            )
            if ep_lines:
                lines.append("Past events: " + "; ".join(ep_lines))
                token_budget -= tokens_used

        # 3. Conversation snippets
        if token_budget > 50:
            conv_lines, tokens_used = self._compress_conversations(
                recall_results.get("conversations", []),
                include_scores=include_scores,
            )
            if conv_lines:
                lines.append("Relevant past: " + "; ".join(conv_lines))

        if not lines:
            return ""

        header = "--- Memory Context ---"
        footer = "--- End Memory ---"
        return f"{header}\n" + "\n".join(lines) + f"\n{footer}"

    # ── Preferences ───────────────────────────────────────────────────────────

    def _compress_preferences(
        self,
        prefs: list[dict],
        include_scores: bool = False,
    ) -> tuple[list[str], int]:
        """Returns (formatted_lines, estimated_tokens)."""
        filtered = [p for p in prefs if p.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="key")
        filtered = filtered[:MAX_PREF_ITEMS]

        lines = []
        for p in filtered:
            key   = self._clean(p.get("key", ""))
            value = self._truncate(self._clean(p.get("value", "")), MAX_VALUE_LEN)
            entry = f"{key}={value}"
            if include_scores:
                entry += f"[{p.get('score', 0):.2f}]"
            lines.append(entry)

        tokens = self._estimate_tokens(" | ".join(lines))
        return lines, tokens

    # ── Episodes ──────────────────────────────────────────────────────────────

    def _compress_episodes(
        self,
        episodes: list[dict],
        include_scores: bool = False,
    ) -> tuple[list[str], int]:
        filtered = [e for e in episodes if e.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="event")
        filtered = filtered[:MAX_EPISODE_ITEMS]

        lines = []
        for ep in filtered:
            event = self._truncate(self._clean(ep.get("event", "")), MAX_EPISODE_LEN)
            ts    = (ep.get("timestamp") or "")[:10]
            entry = f"{event}" + (f" ({ts})" if ts else "")
            if include_scores:
                entry += f"[{ep.get('score', 0):.2f}]"
            lines.append(entry)

        tokens = self._estimate_tokens("; ".join(lines))
        return lines, tokens

    # ── Conversations ─────────────────────────────────────────────────────────

    def _compress_conversations(
        self,
        convos: list[dict],
        include_scores: bool = False,
    ) -> tuple[list[str], int]:
        filtered = [c for c in convos if c.get("score", 0) >= self.threshold]
        filtered = filtered[:MAX_CONVO_ITEMS]

        lines = []
        for c in filtered:
            user   = self._truncate(self._clean(c.get("user_input", "")), MAX_CONVO_LEN)
            assist = self._truncate(self._clean(c.get("assistant_response", "")), MAX_CONVO_LEN)
            entry  = f'"{user}" → "{assist}"'
            if include_scores:
                entry += f"[{c.get('score', 0):.2f}]"
            lines.append(entry)

        tokens = self._estimate_tokens("; ".join(lines))
        return lines, tokens

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Normalize whitespace and strip special chars."""
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~0.75 tokens per character (English average)."""
        return max(1, int(len(text) * 0.75))

    @staticmethod
    def _deduplicate(items: list[dict], key: str) -> list[dict]:
        """Remove duplicate entries by a given key field."""
        seen   = set()
        result = []
        for item in items:
            k = item.get(key, "")
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def explain(self, query: str, recall_results: dict) -> str:
        """
        Return a human-readable explanation of what was included and why.
        Useful for debugging memory recall quality.
        """
        lines = [f"ContextCompressor.explain(query={repr(query[:60])})\n"]

        for category, items in recall_results.items():
            lines.append(f"  [{category.upper()}] — {len(items)} results:")
            for item in items:
                score  = item.get("score", 0)
                source = item.get("source", "?")
                status = "✓ INCLUDED" if score >= self.threshold else "✗ filtered"

                # Get a preview text based on category
                if category == "preferences":
                    preview = f"{item.get('key')}={item.get('value')}"
                elif category == "episodes":
                    preview = item.get("event", "")[:60]
                else:
                    preview = item.get("user_input", "")[:60]

                lines.append(f"    {status} | score={score:.3f} | src={source} | {preview}")

        return "\n".join(lines)

