"""
core/llm/__init__.py
═════════════════════
LLM subsystem for Jarvis — clean 4-component architecture.

Components:
    LLMClientV2    — Public interface, all LLM calls enter here
    OllamaClient   — Local Ollama HTTP client (tried first, no API cost)
    CloudLLMClient — Cloud fallback chain: Gemini → Groq → OpenAI → Anthropic
    ModelRouter    — Maps task_type → model name (pure config, no I/O)
"""

from core.llm.client import LLMClientV2
from core.llm.ollama_client import OllamaClient
from core.llm.cloud_client import CloudLLMClient
from core.llm.model_router import ModelRouter

__all__ = ["LLMClientV2", "OllamaClient", "CloudLLMClient", "ModelRouter"]
