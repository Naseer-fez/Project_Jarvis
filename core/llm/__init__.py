"""
core/llm/__init__.py
═════════════════════
LLM subsystem for Jarvis — adaptive multi-tier architecture.

Components:
    LLMClientV2    — Public interface, all LLM calls enter here
    OllamaClient   — Local Ollama HTTP client (tried first, no API cost)
    CloudLLMClient — Cloud fallback chain: Gemini → Groq → OpenAI → Anthropic
    ModelRouter    — Adaptive task → model routing (cost-optimising)
    ModelSpec      — Model capability/cost descriptors
    ModelRegistry  — Queryable catalog of all known models
    RoutingDecision — Result of an adaptive routing decision
    RoutingTelemetry — Per-model execution stats tracker
"""

from core.llm.client import LLMClientV2
from core.llm.ollama_client import OllamaClient, OllamaTransientError
from core.llm.cloud_client import CloudLLMClient
from core.llm.model_router import ModelRouter
from core.llm.model_spec import ModelSpec, ModelRegistry, RoutingDecision
from core.llm.telemetry import RoutingTelemetry

__all__ = [
    "LLMClientV2",
    "OllamaClient",
    "OllamaTransientError",
    "CloudLLMClient",
    "ModelRouter",
    "ModelSpec",
    "ModelRegistry",
    "RoutingDecision",
    "RoutingTelemetry",
]
