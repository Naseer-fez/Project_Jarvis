"""Model specification registry for adaptive routing.

Each model known to Jarvis is described by a ``ModelSpec`` — a frozen record
of its cost, capabilities and constraints.  ``ModelRegistry`` aggregates
specs and exposes query helpers used by the adaptive ``ModelRouter``.
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Immutable descriptor for a single LLM model."""

    name: str                       # e.g. "mistral:7b", "gemini-2.5-flash"
    provider: str                   # "ollama", "gemini", "groq", "openai", "anthropic"
    tier: int                       # 1 (lightweight) → 3 (heavy reasoning)
    weight: float                   # relative cost weight (0.01 = cheapest local, 1.0 = most expensive cloud)
    max_context_tokens: int = 4096  # context window size
    supports_tools: bool = False    # structured function/tool calling
    supports_vision: bool = False   # multimodal image input
    latency_class: str = "fast"     # "instant" (<200ms), "fast" (<1s), "standard" (<5s), "slow" (>5s)
    reasoning_capability: float = 0.3  # 0.0–1.0 estimated reasoning quality


# ── Built-in model specs ──────────────────────────────────────────────────

_BUILTIN_SPECS: list[ModelSpec] = [
    # ── Ollama Tier 1 — Reflexive ──────────────────────────────────────
    ModelSpec("qwen2.5:0.5b",    "ollama", 1, 0.01, 4096,  False, False, "instant",  0.15),
    ModelSpec("llama3.2:1b",     "ollama", 1, 0.02, 8192,  False, False, "instant",  0.20),
    ModelSpec("qwen2.5:1.5b",    "ollama", 1, 0.02, 4096,  False, False, "instant",  0.20),
    ModelSpec("gemma3:1b",       "ollama", 1, 0.02, 8192,  False, False, "instant",  0.18),
    ModelSpec("gemma2:2b",       "ollama", 1, 0.03, 8192,  False, False, "instant",  0.22),

    # ── Ollama Tier 2 — Execution ──────────────────────────────────────
    ModelSpec("mistral:7b",      "ollama", 2, 0.10, 32768, True,  False, "fast",     0.55),
    ModelSpec("llama3:8b",       "ollama", 2, 0.10, 8192,  True,  False, "fast",     0.55),
    ModelSpec("qwen2.5:7b",      "ollama", 2, 0.10, 32768, True,  False, "fast",     0.55),

    # ── Ollama Tier 3 — Reasoning ──────────────────────────────────────
    ModelSpec("deepseek-r1:8b",  "ollama", 3, 0.15, 32768, True,  False, "standard", 0.75),
    ModelSpec("deepseek-r1:14b", "ollama", 3, 0.25, 32768, True,  False, "standard", 0.82),
    ModelSpec("llama3.3:70b",    "ollama", 3, 0.50, 8192,  True,  False, "slow",     0.88),

    # ── Ollama Vision ──────────────────────────────────────────────────
    ModelSpec("llava",           "ollama", 2, 0.12, 4096,  False, True,  "standard", 0.40),
    ModelSpec("llava:latest",    "ollama", 2, 0.12, 4096,  False, True,  "standard", 0.40),

    # ── Cloud Tier 1 ───────────────────────────────────────────────────
    ModelSpec("gemini-2.0-flash-lite", "gemini",    1, 0.05, 1048576, True,  True,  "fast",     0.45),
    ModelSpec("llama-3.1-8b-instant",  "groq",      1, 0.04, 131072,  True,  False, "instant",  0.40),
    ModelSpec("gpt-4o-mini",           "openai",    1, 0.08, 128000,  True,  True,  "fast",     0.55),
    ModelSpec("claude-3-haiku-20240307","anthropic", 1, 0.06, 200000,  True,  False, "fast",     0.45),

    # ── Cloud Tier 2 ───────────────────────────────────────────────────
    ModelSpec("gemini-2.5-flash",          "gemini",    2, 0.20, 1048576, True,  True,  "fast",     0.80),
    ModelSpec("llama-3.3-70b-versatile",   "groq",      2, 0.15, 131072,  True,  False, "fast",     0.70),
    ModelSpec("gpt-4o",                    "openai",    2, 0.50, 128000,  True,  True,  "standard", 0.82),
    ModelSpec("claude-3-5-sonnet-20241022","anthropic", 2, 0.45, 200000,  True,  False, "standard", 0.85),

    # ── Cloud Tier 3 ───────────────────────────────────────────────────
    ModelSpec("gemini-2.5-pro",                 "gemini",    3, 0.60, 1048576, True,  True,  "standard", 0.92),
    ModelSpec("deepseek-r1-distill-llama-70b",  "groq",      3, 0.30, 131072,  True,  False, "standard", 0.80),
    ModelSpec("o3-mini",                        "openai",    3, 0.70, 128000,  True,  False, "standard", 0.90),
    ModelSpec("claude-sonnet-4-20250514",       "anthropic", 3, 0.80, 200000,  True,  False, "standard", 0.93),
]


@dataclass(frozen=True)
class RoutingDecision:
    """Result of an adaptive routing decision — carries the chosen model + rationale."""

    model: str
    provider: str
    tier: int
    reason: str
    weight: float = 0.0
    fallback_model: str | None = None
    fallback_provider: str | None = None


class ModelRegistry:
    """Queryable catalog of all known model specs.

    Loads built-in defaults, then overlays any ``[models.registry]`` section
    from the Jarvis config file.
    """

    def __init__(self, config: configparser.ConfigParser | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in _BUILTIN_SPECS:
            self._specs[spec.name] = spec

        # Overlay config overrides
        if config is not None:
            self._load_config_overrides(config)

    # ── Public queries ─────────────────────────────────────────────────

    def get(self, model_name: str) -> ModelSpec | None:
        """Look up a spec by exact name, or by family prefix."""
        spec = self._specs.get(model_name)
        if spec is not None:
            return spec
        # Try family match (e.g. "mistral" → "mistral:7b")
        family = model_name.split(":")[0]
        for name, s in self._specs.items():
            if name.split(":")[0] == family:
                return s
        return None

    def get_tier(self, model_name: str) -> int:
        """Return the tier for a model, defaulting to 2."""
        spec = self.get(model_name)
        return spec.tier if spec else 2

    def get_weight(self, model_name: str) -> float:
        """Return the cost weight for a model, defaulting to 0.5."""
        spec = self.get(model_name)
        return spec.weight if spec else 0.5

    def all_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def by_provider(self, provider: str) -> list[ModelSpec]:
        return [s for s in self._specs.values() if s.provider == provider]

    def by_tier(self, tier: int) -> list[ModelSpec]:
        return sorted(
            [s for s in self._specs.values() if s.tier == tier],
            key=lambda s: s.weight,
        )

    def get_cheapest_capable(
        self,
        *,
        min_reasoning: float = 0.0,
        min_context: int = 0,
        needs_tools: bool = False,
        needs_vision: bool = False,
        available_models: set[str] | None = None,
        providers: set[str] | None = None,
    ) -> list[ModelSpec]:
        """Return models meeting the requirements, sorted cheapest-first.

        Parameters
        ----------
        available_models:
            If provided, restrict to these exact model names (for Ollama
            availability filtering).  Cloud models are included regardless
            unless *providers* is set.
        providers:
            If provided, restrict to these providers.
        """
        candidates: list[ModelSpec] = []
        for spec in self._specs.values():
            if spec.reasoning_capability < min_reasoning:
                continue
            if spec.max_context_tokens < min_context:
                continue
            if needs_tools and not spec.supports_tools:
                continue
            if needs_vision and not spec.supports_vision:
                continue
            if providers and spec.provider not in providers:
                continue
            # For ollama models, check availability
            if spec.provider == "ollama" and available_models is not None:
                if spec.name not in available_models:
                    continue
            candidates.append(spec)

        return sorted(candidates, key=lambda s: (s.weight, -s.reasoning_capability))

    def register(self, spec: ModelSpec) -> None:
        """Add or replace a model spec."""
        self._specs[spec.name] = spec

    # ── Config overlay ─────────────────────────────────────────────────

    def _load_config_overrides(self, config: configparser.ConfigParser) -> None:
        """Parse ``[models.registry]`` entries like ``mistral:7b = tier=2,weight=0.10``."""
        section = "models.registry"
        if not config.has_section(section):
            return
        for model_name in config.options(section):
            raw = config.get(section, model_name, fallback="")
            if not raw.strip():
                continue
            try:
                overrides = self._parse_override(raw)
                existing = self._specs.get(model_name)
                if existing:
                    # Merge overrides into existing spec via replace
                    kwargs = {f.name: getattr(existing, f.name) for f in existing.__dataclass_fields__.values()}
                    kwargs.update(overrides)
                    self._specs[model_name] = ModelSpec(**kwargs)
                else:
                    # Require at least provider for new specs
                    if "provider" not in overrides:
                        logger.warning("Skipping model %s: no provider specified", model_name)
                        continue
                    overrides.setdefault("name", model_name)
                    overrides.setdefault("tier", 2)
                    overrides.setdefault("weight", 0.5)
                    self._specs[model_name] = ModelSpec(**overrides)
            except Exception as exc:
                logger.warning("Failed to parse model registry override for %s: %s", model_name, exc)

    @staticmethod
    def _parse_override(raw: str) -> dict[str, Any]:
        """Parse ``tier=2,weight=0.10,supports_tools=true`` into a dict."""
        result: dict[str, Any] = {}
        for pair in raw.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Type coercion
            if key in ("tier", "max_context_tokens"):
                result[key] = int(value)
            elif key in ("weight", "reasoning_capability"):
                result[key] = float(value)
            elif key in ("supports_tools", "supports_vision"):
                result[key] = value.lower() in ("true", "1", "yes")
            else:
                result[key] = value
        return result


__all__ = ["ModelSpec", "ModelRegistry", "RoutingDecision"]
