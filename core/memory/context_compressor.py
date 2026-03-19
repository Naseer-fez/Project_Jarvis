"""
Context compression and optional low-latency focus titling for Jarvis memory.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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

    def compress(
        self,
        query: str,
        recall_results: dict,
        include_scores: bool = False,
    ) -> str:
        """Build a compact text block from structured recall results."""
        lines: list[str] = []
        token_budget = self.max_tokens

        pref_lines, tokens_used = self._compress_preferences(
            recall_results.get("preferences", []),
            include_scores,
        )
        if pref_lines:
            lines.append("Preferences: " + " | ".join(pref_lines))
            token_budget -= tokens_used

        if token_budget > 50:
            episode_lines, tokens_used = self._compress_episodes(
                recall_results.get("episodes", []),
                include_scores,
            )
            if episode_lines:
                lines.append("Past events: " + "; ".join(episode_lines))
                token_budget -= tokens_used

        if token_budget > 50:
            convo_lines, _ = self._compress_conversations(
                recall_results.get("conversations", []),
                include_scores,
            )
            if convo_lines:
                lines.append("Relevant past: " + "; ".join(convo_lines))

        if not lines:
            return ""

        title = self._generate_focus_title(query, lines)
        if title:
            lines.insert(0, f"Focus: {title}")

        return "--- Memory Context ---\n" + "\n".join(lines) + "\n--- End Memory ---"

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

    def _generate_focus_title(self, query: str, lines: list[str]) -> str:
        """Generate a short semantic title using the quick-task LLM route."""
        if not self.enable_llm_title or self.llm is None or not lines:
            return ""

        prompt = (
            f"User query: {self._clean(query)}\n"
            f"Memory context:\n{chr(10).join(lines[:3])}\n\n"
            "Return the best short title for this memory context."
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.llm.complete(
                        prompt,
                        system=TITLE_SYSTEM,
                        temperature=0.0,
                        task_type="context_title_generation",
                    ),
                )
                raw = str(future.result(timeout=TITLE_TIMEOUT_S) or "")
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
