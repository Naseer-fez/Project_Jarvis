"""
Context compression and optional low-latency focus titling for Jarvis memory.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 400
DEFAULT_THRESHOLD = 0.30
MAX_PREF_ITEMS = 6
MAX_EPISODE_ITEMS = 3
MAX_CONVO_ITEMS = 2
MAX_VALUE_LEN = 60
MAX_EPISODE_LEN = 100
MAX_CONVO_LEN = 120
TITLE_TIMEOUT_S = 4.0
TITLE_SYSTEM = (
    "You create short memory context titles for a local AI assistant. "
    "Return only a 3-7 word title. No quotes, no bullets, no punctuation-heavy output."
)


class ContextCompressor:
    """Compress recalled memory into a compact block for LLM injection."""

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        threshold: float = DEFAULT_THRESHOLD,
        llm: Any | None = None,
        enable_llm_title: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.llm = llm
        self.enable_llm_title = bool(enable_llm_title)

    async def compress(
        self,
        query: str,
        recall_results: dict,
        include_scores: bool = False,
    ) -> str:
        """Build a compact text block from structured recall results with aging and deduplication."""
        import hashlib
        import math
        from datetime import datetime

        # 1. Gather and deduplicate all items
        all_items = []
        seen_hashes = set()
        
        for category, items in recall_results.items():
            for item in items:
                item_copy = dict(item)
                item_copy["_category"] = category
                text = self._get_item_text(item_copy)
                h = hashlib.md5(text.encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_items.append(item_copy)

        # 2. Apply temporal memory aging (decay similarity scores by e^(-0.05 * t))
        decayed_items = []
        for item in all_items:
            score = float(item.get("score", 1.0))
            ts = item.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    now = datetime.now(dt.tzinfo)
                    age_days = max(0.0, (now - dt).total_seconds() / (24 * 3600))
                    decay = math.exp(-0.05 * age_days)
                    score *= decay
                except Exception:
                    pass
            item["score"] = score
            decayed_items.append(item)

        # 3. Sort by decayed score and keep top 5
        decayed_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top_5 = decayed_items[:5]

        if not top_5:
            return ""

        # 4. Check if combined text length exceeds limit, and summarize
        combined_text = "\n".join([self._get_item_text(item) for item in top_5])
        if self.llm and self._estimate_tokens(combined_text) > self.max_tokens:
            summary = await self._summarize_context(top_5)
            if summary:
                return f"--- Memory Context ---\nMemory Summary: {summary}\n--- End Memory ---"

        # 5. Format the top 5 high-score entries under categories as fallback
        prefs = [item for item in top_5 if item["_category"] == "preferences"]
        episodes = [item for item in top_5 if item["_category"] == "episodes"]
        convos = [item for item in top_5 if item["_category"] == "conversations"]

        lines: list[str] = []
        token_budget = self.max_tokens

        pref_lines, tokens_used = self._compress_preferences(prefs, include_scores)
        if pref_lines:
            lines.append("Preferences: " + " | ".join(pref_lines))
            token_budget -= tokens_used

        if token_budget > 50:
            episode_lines, tokens_used = self._compress_episodes(episodes, include_scores)
            if episode_lines:
                lines.append("Past events: " + "; ".join(episode_lines))
                token_budget -= tokens_used

        if token_budget > 50:
            convo_lines, _ = self._compress_conversations(convos, include_scores)
            if convo_lines:
                lines.append("Relevant past: " + "; ".join(convo_lines))

        if not lines:
            return ""

        title = await self._generate_focus_title(query, lines)
        if title:
            lines.insert(0, f"Focus: {title}")

        return "--- Memory Context ---\n" + "\n".join(lines) + "\n--- End Memory ---"

    def _get_item_text(self, item: dict) -> str:
        if "key" in item and "value" in item and item["key"]:
            return f"Preference: {item['key']}={item['value']}"
        elif "event" in item:
            return f"Event: {item['event']}"
        elif "user_input" in item:
            return f"Conversation: User: {item['user_input']} -> Assistant: {item['assistant_response']}"
        return str(item.get("document", ""))

    async def _summarize_context(self, top_memories: list[dict]) -> str:
        if not self.llm or not top_memories:
            return ""
        
        memory_strings = [self._get_item_text(item) for item in top_memories]
        context_text = "\n".join(memory_strings)
        prompt = (
            "You are a helpful context summarizer for a personal AI assistant.\n"
            "Summarize the following retrieved memories into a short, cohesive summary (less than 150 words) "
            "that highlights key facts, preferences, and relevant history:\n\n"
            f"{context_text}\n\n"
            "Summary:"
        )
        try:
            summary = await self.llm.complete(
                prompt,
                system="Summarize the assistant's memory context accurately.",
                temperature=0.0,
                task_type="context_summarization",
            )
            return summary.strip()
        except Exception as exc:
            logger.debug("Failed to summarize context with LLM: %s", exc)
            return ""

    def _compress_preferences(
        self,
        prefs: list[dict],
        include_scores: bool,
    ) -> tuple[list[str], int]:
        filtered = [item for item in prefs if item.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="key")[:MAX_PREF_ITEMS]
        lines: list[str] = []

        for item in filtered:
            key = self._clean(item.get("key", ""))
            value = self._truncate(self._clean(item.get("value", "")), MAX_VALUE_LEN)
            entry = f"{key}={value}"
            if include_scores:
                entry += f"[{item.get('score', 0):.2f}]"
            lines.append(entry)

        return lines, self._estimate_tokens(" | ".join(lines))

    def _compress_episodes(
        self,
        episodes: list[dict],
        include_scores: bool,
    ) -> tuple[list[str], int]:
        filtered = [item for item in episodes if item.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="event")[:MAX_EPISODE_ITEMS]
        lines: list[str] = []

        for item in filtered:
            event = self._truncate(self._clean(item.get("event", "")), MAX_EPISODE_LEN)
            timestamp = (item.get("timestamp") or "")[:10]
            entry = f"{event}" + (f" ({timestamp})" if timestamp else "")
            if include_scores:
                entry += f"[{item.get('score', 0):.2f}]"
            lines.append(entry)

        return lines, self._estimate_tokens("; ".join(lines))

    def _compress_conversations(
        self,
        convos: list[dict],
        include_scores: bool,
    ) -> tuple[list[str], int]:
        filtered = [item for item in convos if item.get("score", 0) >= self.threshold]
        filtered = filtered[:MAX_CONVO_ITEMS]
        lines: list[str] = []

        for item in filtered:
            user = self._truncate(self._clean(item.get("user_input", "")), MAX_CONVO_LEN)
            assistant = self._truncate(
                self._clean(item.get("assistant_response", "")),
                MAX_CONVO_LEN,
            )
            entry = f'"{user}" -> "{assistant}"'
            if include_scores:
                entry += f"[{item.get('score', 0):.2f}]"
            lines.append(entry)

        return lines, self._estimate_tokens("; ".join(lines))

    async def _generate_focus_title(self, query: str, lines: list[str]) -> str:
        """Generate a short semantic title using the quick-task LLM route."""
        if not self.enable_llm_title or self.llm is None or not lines:
            return ""

        prompt = (
            f"User query: {self._clean(query)}\n"
            f"Memory context:\n{chr(10).join(lines[:3])}\n\n"
            "Return the best short title for this memory context."
        )

        try:
            raw = await asyncio.wait_for(
                self.llm.complete(
                    prompt,
                    system=TITLE_SYSTEM,
                    temperature=0.0,
                    task_type="context_title_generation",
                ),
                timeout=TITLE_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Context title generation failed: %s", exc)
            return ""

        title = self._clean(raw.splitlines()[0] if raw else "")
        title = title.strip("\"'` .")
        if not title:
            return ""
        return self._truncate(title, 48)

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) * 0.75))

    @staticmethod
    def _deduplicate(items: list[dict], key: str) -> list[dict]:
        seen: set[str] = set()
        result: list[dict] = []
        for item in items:
            value = str(item.get(key, ""))
            if value in seen:
                continue
            seen.add(value)
            result.append(item)
        return result

    def explain(self, query: str, recall_results: dict) -> str:
        """Return a human-readable explanation of what would be included."""
        lines = [f"ContextCompressor.explain(query={query[:60]!r})\n"]
        for category, items in recall_results.items():
            lines.append(f"  [{str(category).upper()}] - {len(items)} results:")
            for item in items:
                lines.append(f"    INCLUDED | score={item.get('score', 0):.3f}")
        return "\n".join(lines)
