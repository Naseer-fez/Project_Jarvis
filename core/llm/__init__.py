"""
core/llm/__init__.py
═════════════════════
LLM client for Jarvis V1 — DeepSeek R1:8b via Ollama.

Exposes: LLMClientV2
- Async, non-blocking
- Strips <think>...</think> blocks from DeepSeek R1 output
- Never calls cloud APIs
- Returns empty string on failure (never raises)
"""

from core.llm.client import LLMClientV2

__all__ = ["LLMClientV2"]