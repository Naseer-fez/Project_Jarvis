"""Heuristics-based complexity scorer for Adaptive Intelligence Routing."""

from __future__ import annotations

import re
import logging
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets (unchanged categories)
# ---------------------------------------------------------------------------
_REFLEX_KEYWORDS = {"weather", "time", "date", "status", "hello", "hi", "ping"}

_DEEP_KEYWORDS = {
    "architecture", "debug", "refactor", "complex", "system design",
    "explain how", "why is this failing", "optimize",
}

_AGENTIC_KEYWORDS = {
    "create", "write", "plan", "workflow", "automate", "search", "find",
    "organize", "download", "fetch", "open", "launch", "start", "close",
    "type", "click", "do", "execute", "run", "make",
}

_CONDITIONAL_WORDS = {"if", "when", "unless", "assuming", "provided", "suppose"}
_TECHNICAL_TERMS = {
    "api", "async", "await", "class", "function", "method", "endpoint",
    "database", "schema", "deploy", "container", "docker", "kubernetes",
    "pipeline", "microservice", "oauth", "jwt", "websocket", "regex",
}

# Token-estimation multipliers per category
_TOKEN_MULTIPLIERS: dict[str, float] = {
    "Reflex": 1.5,
    "Chat": 4.0,
    "Agentic": 6.0,
    "Deep_Reasoning": 10.0,
}


# ---------------------------------------------------------------------------
# Structural / vocabulary helpers
# ---------------------------------------------------------------------------
def _structural_signals(text: str) -> dict[str, Any]:
    """Extract structural signals from the raw input."""
    words = text.split()
    word_count = len(words)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    question_marks = text.count("?")
    has_code = "```" in text or bool(re.search(r'`[^`]+`', text))
    has_vision = bool(re.search(r'\.(png|jpg|jpeg|gif|bmp|webp|svg)\b', text, re.I))

    # Multi-part detection: numbered lists, bullets, conjunctions
    multi_part = bool(re.search(r'(\d+[.)]\s)|([•\-\*]\s)|(,\s*and\s)|\balso\b|\bthen\b', text))

    # Technical term density (fraction of words that are technical)
    lower_words = [w.strip(".,;:!?\"'()[]{}") for w in words]
    tech_count = sum(1 for w in lower_words if w in _TECHNICAL_TERMS)
    tech_density = tech_count / max(word_count, 1)

    # Conditional word count
    cond_count = sum(1 for w in lower_words if w in _CONDITIONAL_WORDS)

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "question_marks": question_marks,
        "has_code": has_code,
        "has_vision": has_vision,
        "multi_part": multi_part,
        "tech_density": tech_density,
        "conditional_count": cond_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_request(user_input: str) -> dict[str, Any]:
    """Classify the complexity and type of request to determine routing.

    Classes: Reflex, Chat, Agentic, Deep_Reasoning.

    Returns a dict with routing metadata *and* enriched signals:
    ``class``, ``complexity``, ``route``, ``skip_planner``,
    ``estimated_tokens``, ``needs_reasoning``, ``needs_tools``,
    ``needs_vision``, ``context_weight``.
    """
    text = user_input.lower().strip()
    sig = _structural_signals(text)

    # --- 1. Base classification (keyword matching) ------------------------
    if text in _REFLEX_KEYWORDS or any(
        text.startswith(f"what is the {k}") or text.startswith(f"whats the {k}")
        or text.startswith("what time")
        for k in _REFLEX_KEYWORDS
    ):
        cls, base_cx, route, skip = "Reflex", 0.1, "direct", True
    elif any(k in text for k in _DEEP_KEYWORDS):
        cls, base_cx, route, skip = "Deep_Reasoning", 0.9, "premium", False
    elif any(k in text for k in _AGENTIC_KEYWORDS) or text in {"do it", "go"}:
        cls, base_cx, route, skip = "Agentic", 0.6, "planner", False
    else:
        cls, base_cx, route, skip = "Chat", 0.4, "mid-tier", True

    # --- 2. Complexity modifiers ------------------------------------------
    cx = base_cx
    if sig["word_count"] > 200:
        cx += 0.2
    if sig["multi_part"]:
        cx += 0.15
    if sig["has_code"]:
        cx += 0.1
    # Extra nudge for heavy conditional / technical language
    if sig["tech_density"] > 0.15:
        cx += 0.05
    if sig["conditional_count"] >= 2:
        cx += 0.05
    cx = max(0.0, min(cx, 1.0))

    # --- 3. Derived signals -----------------------------------------------
    estimated_tokens = int(sig["word_count"] * _TOKEN_MULTIPLIERS.get(cls, 4.0))
    needs_reasoning = cls == "Deep_Reasoning" or cx >= 0.75
    needs_tools = cls == "Agentic"
    context_weight = round(min(1.0, 0.2 + sig["tech_density"] + 0.1 * sig["sentence_count"]), 2)

    return {
        "class": cls,
        "complexity": round(cx, 2),
        "route": route,
        "skip_planner": skip,
        "estimated_tokens": estimated_tokens,
        "needs_reasoning": needs_reasoning,
        "needs_tools": needs_tools,
        "needs_vision": sig["has_vision"],
        "context_weight": context_weight,
    }


__all__ = ["classify_request"]
