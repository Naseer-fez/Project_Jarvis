"""Heuristics-based complexity scorer for Adaptive Intelligence Routing."""

from __future__ import annotations

def classify_request(user_input: str) -> dict[str, str | float | bool]:
    """Classify the complexity and type of request to determine routing.
    
    Classes: Reflex, Tool-only, Chat, Agentic, Deep_Reasoning, Autonomous
    """
    text = user_input.lower().strip()
    
    # 1. Reflex (Low complexity, direct response or very fast path)
    reflex_keywords = {"weather", "time", "date", "status", "hello", "hi", "ping"}
    # Exact match or simple starts-with
    if text in reflex_keywords or any(text.startswith(f"what is the {k}") or text.startswith(f"whats the {k}") or text.startswith(f"what time") for k in reflex_keywords):
        return {"class": "Reflex", "complexity": 0.1, "skip_planner": True, "route": "direct"}

    # 2. Tool-only (Low complexity, direct executor)
    tool_only_keywords = {"open", "launch", "start", "close"}
    # Very short commands like "open notepad"
    if any(text.startswith(k) for k in tool_only_keywords) and len(text.split()) <= 5:
        return {"class": "Tool-only", "complexity": 0.2, "skip_planner": True, "route": "executor"}
        
    # 3. Deep Reasoning (High complexity, premium LLM)
    deep_keywords = {"architecture", "debug", "refactor", "complex", "system design", "explain how", "why is this failing", "optimize"}
    if any(k in text for k in deep_keywords):
        return {"class": "Deep_Reasoning", "complexity": 0.9, "skip_planner": False, "route": "premium"}

    # 4. Agentic (Medium complexity, planner + DAG)
    agentic_keywords = {"create", "write", "plan", "workflow", "automate", "search", "find", "organize", "download", "fetch"}
    if any(k in text for k in agentic_keywords) and len(text.split()) > 2:
        return {"class": "Agentic", "complexity": 0.6, "skip_planner": False, "route": "planner"}

    # 5. Default Chat (Low/Medium complexity, mid-tier LLM)
    return {"class": "Chat", "complexity": 0.4, "skip_planner": True, "route": "mid-tier"}

__all__ = ["classify_request"]
