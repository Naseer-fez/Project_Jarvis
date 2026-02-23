"""
core/context_compressor.py
───────────────────────────
Context compression and memory injection for Jarvis.
"""

import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS  = 400
DEFAULT_THRESHOLD   = 0.30
MAX_PREF_ITEMS      = 6
MAX_EPISODE_ITEMS   = 3
MAX_CONVO_ITEMS     = 2
MAX_VALUE_LEN       = 60
MAX_EPISODE_LEN     = 100
MAX_CONVO_LEN       = 120

class ContextCompressor:
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, threshold: float = DEFAULT_THRESHOLD):
        self.max_tokens = max_tokens
        self.threshold  = threshold

    def compress(self, query: str, recall_results: dict, include_scores: bool = False) -> str:
        lines = []
        token_budget = self.max_tokens

        pref_lines, t_used = self._compress_preferences(recall_results.get("preferences", []), include_scores)
        if pref_lines:
            lines.append("Preferences: " + " | ".join(pref_lines))
            token_budget -= t_used

        if token_budget > 50:
            ep_lines, t_used = self._compress_episodes(recall_results.get("episodes", []), include_scores)
            if ep_lines:
                lines.append("Past events: " + "; ".join(ep_lines))
                token_budget -= t_used

        if token_budget > 50:
            conv_lines, t_used = self._compress_conversations(recall_results.get("conversations", []), include_scores)
            if conv_lines:
                lines.append("Relevant past: " + "; ".join(conv_lines))

        if not lines:
            return ""

        return f"--- Memory Context ---\n" + "\n".join(lines) + f"\n--- End Memory ---"

    def _compress_preferences(self, prefs: list[dict], include_scores: bool) -> tuple[list[str], int]:
        filtered = [p for p in prefs if p.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="key")[:MAX_PREF_ITEMS]
        lines = []
        for p in filtered:
            k = self._clean(p.get("key", ""))
            v = self._truncate(self._clean(p.get("value", "")), MAX_VALUE_LEN)
            entry = f"{k}={v}"
            if include_scores: entry += f"[{p.get('score', 0):.2f}]"
            lines.append(entry)
        return lines, self._estimate_tokens(" | ".join(lines))

    def _compress_episodes(self, episodes: list[dict], include_scores: bool) -> tuple[list[str], int]:
        filtered = [e for e in episodes if e.get("score", 0) >= self.threshold]
        filtered = self._deduplicate(filtered, key="event")[:MAX_EPISODE_ITEMS]
        lines = []
        for ep in filtered:
            event = self._truncate(self._clean(ep.get("event", "")), MAX_EPISODE_LEN)
            ts = (ep.get("timestamp") or "")[:10]
            entry = f"{event}" + (f" ({ts})" if ts else "")
            if include_scores: entry += f"[{ep.get('score', 0):.2f}]"
            lines.append(entry)
        return lines, self._estimate_tokens("; ".join(lines))

    def _compress_conversations(self, convos: list[dict], include_scores: bool) -> tuple[list[str], int]:
        filtered = [c for c in convos if c.get("score", 0) >= self.threshold][:MAX_CONVO_ITEMS]
        lines = []
        for c in filtered:
            user = self._truncate(self._clean(c.get("user_input", "")), MAX_CONVO_LEN)
            assist = self._truncate(self._clean(c.get("assistant_response", "")), MAX_CONVO_LEN)
            entry = f'"{user}" → "{assist}"'
            if include_scores: entry += f"[{c.get('score', 0):.2f}]"
            lines.append(entry)
        return lines, self._estimate_tokens("; ".join(lines))

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[:max_len - 3] + "..."

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, int(len(text) * 0.75))

    @staticmethod
    def _deduplicate(items: list[dict], key: str) -> list[dict]:
        seen, result = set(), []
        for item in items:
            k = item.get(key, "")
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result

    def explain(self, query: str, recall_results: dict) -> str:
        lines = [f"ContextCompressor.explain(query={repr(query[:60])})\n"]
        for category, items in recall_results.items():
            lines.append(f"  [{category.upper()}] — {len(items)} results:")
            for item in items:
                lines.append(f"    ✓ INCLUDED | score={item.get('score', 0):.3f}")
        return "\n".join(lines)

