"""Intelligent multi-stage model router for Jarvis.

Supports two strategies:
  - ``adaptive`` (default): cost-minimising selection based on task
    classification signals, model capabilities, availability, and
    historical reliability from telemetry.
  - ``static``: legacy tier-lookup behaviour (backward compatible).

Public surface (all preserved from the original):
  - ``route(task_type)``
  - ``pick_model(task_type)``
  - ``get_best_available(task_type)``
  - ``escalate(current_model)``

New:
  - ``route_adaptive(classification)`` → ``RoutingDecision``
  - ``should_escalate(model, task_type, response)`` → bool
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Iterable, Any

from core.config.defaults import OLLAMA_BASE_URL
from core.llm.ollama_client import list_models_sync
from core.llm.model_spec import ModelSpec, ModelRegistry, RoutingDecision

logger = logging.getLogger(__name__)

_MODEL_DISCOVERY_TTL_S = 30.0

# ── Minimum-reasoning thresholds by complexity band ────────────────────
_REASONING_BY_COMPLEXITY = [
    # (max_complexity, min_reasoning_required)
    (0.15, 0.10),   # reflex
    (0.40, 0.25),   # light chat
    (0.60, 0.40),   # standard chat / agentic
    (0.80, 0.60),   # complex
    (1.01, 0.75),   # deep reasoning
]


class ModelRouter:
    """Intelligent multi-stage model router for Jarvis."""

    # ── Legacy tier lists (kept for static mode & backward compat) ─────
    TIERS = {
        1: ["llama3.2:1b", "qwen2.5:0.5b", "qwen2.5:1.5b", "gemma2:2b"],
        2: ["mistral:7b", "llama3:8b", "qwen2.5:7b"],
        3: ["deepseek-r1:8b", "deepseek-r1:14b", "llama3.3:70b"],
    }

    TASK_TIERS = {
        "intent": 1,
        "intent_classification": 1,
        "reflex": 1,
        "web_search_summary": 1,
        "tool_parameter_extraction": 2,
        "context_title_generation": 1,
        "synthesis": 1,
        "summarize": 1,
        "memory_summarization": 1,
        "context_summarization": 1,

        "chat": 2,
        "general": 2,
        "planning": 3,
        "plan": 3,
        "tool_selection": 2,
        "tool_picker": 2,
        "reflection": 2,

        "deep_reasoning": 3,
        "complex_debugging": 3,
        "architecture_generation": 3,
    }

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._available_ollama_models: set[str] = set()
        self._initial_fetch_done = False
        self._cache_time = 0.0
        self._lock = threading.RLock()
        self._telemetry: Any = None  # set via set_telemetry()

        # ── Strategy ───────────────────────────────────────────────────
        self._strategy = "static"
        self._confidence_threshold = 0.7
        self._max_escalations = 1
        self._cost_preference = "balanced"  # minimum | balanced | quality

        # ── Model registry ─────────────────────────────────────────────
        self._registry = ModelRegistry(config)

        # Override TIERS from config if provided (legacy support)
        if self.config:
            if hasattr(self.config, "has_section") and self.config.has_section("routing"):
                self._strategy = self.config.get("routing", "strategy", fallback="static").strip().lower()
                self._confidence_threshold = float(
                    self.config.get("routing", "confidence_threshold", fallback="0.7")
                )
                self._max_escalations = int(
                    self.config.get("routing", "max_escalations", fallback="1")
                )
                self._cost_preference = str(
                    self.config.get("routing", "cost_preference", fallback="balanced")
                ).strip().lower()

                for tier_num in (1, 2, 3):
                    val = self.config.get("routing", f"tier{tier_num}", fallback="")
                    if val:
                        self.TIERS[tier_num] = [m.strip() for m in val.split(",") if m.strip()]
            elif isinstance(self.config, dict) and "routing" in self.config:
                routing = self.config["routing"]
                if isinstance(routing, dict):
                    self._strategy = str(routing.get("strategy", "static")).strip().lower()
                    self._confidence_threshold = float(routing.get("confidence_threshold", "0.7"))
                    self._max_escalations = int(routing.get("max_escalations", "1"))
                    self._cost_preference = str(routing.get("cost_preference", "balanced")).strip().lower()
                    
                    for tier_num in (1, 2, 3):
                        val = str(routing.get(f"tier{tier_num}", ""))
                        if val:
                            self.TIERS[tier_num] = [m.strip() for m in val.split(",") if m.strip()]

    # ── Telemetry wiring ──────────────────────────────────────────────

    def set_telemetry(self, telemetry: Any) -> None:
        """Connect execution telemetry for adaptive routing feedback."""
        self._telemetry = telemetry

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def strategy(self) -> str:
        return self._strategy

    # ── Public routing API (backward compatible) ──────────────────────

    def route(self, task_type: str) -> str:
        """Determines the target tier for a task and returns the best available model."""
        task = str(task_type or "chat").strip().lower()

        # Vision is a special case
        if task == "vision":
            return self._cfg("vision_model", "llava:latest")

        if self._strategy == "adaptive":
            # Build a minimal classification from task_type
            tier = self.TASK_TIERS.get(task, 2)
            classification = {
                "complexity": {1: 0.1, 2: 0.4, 3: 0.85}.get(tier, 0.4),
                "needs_reasoning": tier >= 3,
                "needs_tools": task in ("tool_picker", "tool_selection", "tool_parameter_extraction"),
                "needs_vision": False,
                "estimated_tokens": {1: 50, 2: 200, 3: 500}.get(tier, 200),
                "context_weight": {1: 0.0, 2: 0.3, 3: 0.6}.get(tier, 0.3),
            }
            decision = self.route_adaptive(classification)
            return decision.model

        # Static fallback
        target_tier = self.TASK_TIERS.get(task, 2)
        return self._pick_model_from_tier(target_tier)

    def escalate(self, current_model: str) -> str:
        """Upgrades the model to a higher tier if possible."""
        current_tier = self._registry.get_tier(current_model)
        next_tier = min(3, current_tier + 1)

        if self._strategy == "adaptive":
            # Pick the cheapest model at the next tier that's available
            candidates = self._registry.get_cheapest_capable(
                min_reasoning=0.0,
                available_models=self._available_ollama_models or None,
            )
            for spec in candidates:
                if spec.tier >= next_tier:
                    resolved = self._resolve_available_variant(spec.name)
                    if resolved or spec.provider != "ollama":
                        return resolved or spec.name
            # Fallback
            return self._pick_model_from_tier(next_tier)

        return self._pick_model_from_tier(next_tier)

    def pick_model(self, task_type: str) -> str:
        """Public entry point — returns the best model name for a task type."""
        return self.route(task_type)

    def get_best_available(self, task_type: str) -> str:
        return self.route(task_type)

    # ── Adaptive routing (new) ────────────────────────────────────────

    def route_adaptive(self, classification: dict[str, Any]) -> RoutingDecision:
        """Cost-optimising model selection using classification signals.

        Algorithm:
            1. Compute minimum capability requirements from classification
            2. Query ModelRegistry for models meeting requirements
            3. Filter by availability (Ollama running, cloud API key present)
            4. Sort by weight (cost) ascending, reliability descending
            5. Select cheapest model with reliability ≥ threshold
            6. If none qualifies, escalate tier
        """
        complexity = float(classification.get("complexity", 0.4))
        needs_reasoning = bool(classification.get("needs_reasoning", False))
        needs_tools = bool(classification.get("needs_tools", False))
        needs_vision = bool(classification.get("needs_vision", False))
        estimated_tokens = int(classification.get("estimated_tokens", 200))

        # 1. Compute minimum reasoning from complexity
        min_reasoning = 0.25  # baseline
        for threshold, required in _REASONING_BY_COMPLEXITY:
            if complexity <= threshold:
                min_reasoning = required
                break

        # Boost for explicit reasoning need
        if needs_reasoning:
            min_reasoning = max(min_reasoning, 0.60)

        # Cost preference adjusts the reasoning floor
        if self._cost_preference == "minimum":
            min_reasoning = max(0.10, min_reasoning - 0.15)
        elif self._cost_preference == "quality":
            min_reasoning = min(1.0, min_reasoning + 0.15)

        # 2. Estimated context (input tokens × 1.5 safety margin)
        min_context = int(estimated_tokens * 1.5)

        # 3. Determine available providers
        available_providers = {"ollama"}
        _cloud_keys = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        for provider, env_key in _cloud_keys.items():
            if os.environ.get(env_key):
                available_providers.add(provider)

        # 4. Query registry
        with self._lock:
            self.refresh_available_models()
            candidates = self._registry.get_cheapest_capable(
                min_reasoning=min_reasoning,
                min_context=min_context,
                needs_tools=needs_tools,
                needs_vision=needs_vision,
                available_models=self._available_ollama_models or None,
                providers=available_providers,
            )

        if not candidates:
            # Nothing matches — fall back to static routing
            logger.debug("Adaptive routing found no candidates; falling back to static")
            tier = 2 if complexity < 0.7 else 3
            model = self._pick_model_from_tier(tier)
            return RoutingDecision(
                model=model,
                provider="ollama",
                tier=tier,
                reason="fallback_no_candidates",
            )

        # 5. Apply telemetry-based reliability filtering
        selected = candidates[0]  # cheapest by default
        if self._telemetry is not None:
            for spec in candidates:
                reliability = self._telemetry.get_reliability(spec.name, "overall")
                if reliability >= self._confidence_threshold:
                    selected = spec
                    break
            else:
                # No model meets threshold — pick the most reliable
                best_reliability = -1.0
                for spec in candidates:
                    rel = self._telemetry.get_reliability(spec.name, "overall")
                    if rel > best_reliability:
                        best_reliability = rel
                        selected = spec

        # 6. Determine fallback (next cheapest at higher tier)
        fallback_model = None
        fallback_provider = None
        for spec in candidates:
            if spec.tier > selected.tier and spec.name != selected.name:
                fallback_model = spec.name
                fallback_provider = spec.provider
                break

        reason = (
            f"adaptive: complexity={complexity:.2f}, "
            f"min_reasoning={min_reasoning:.2f}, "
            f"cost_pref={self._cost_preference}, "
            f"candidates={len(candidates)}"
        )
        logger.info(
            "Routing decision: %s (tier %d, weight %.2f) — %s",
            selected.name, selected.tier, selected.weight, reason,
        )

        return RoutingDecision(
            model=selected.name,
            provider=selected.provider,
            tier=selected.tier,
            reason=reason,
            weight=selected.weight,
            fallback_model=fallback_model,
            fallback_provider=fallback_provider,
        )

    def should_escalate(self, model: str, task_type: str, response: str) -> bool:
        """Heuristic check if a response quality is too low and needs retry."""
        if not response or not response.strip():
            return True

        text = response.strip()

        # Suspiciously short response for non-reflex tasks
        tier = self.TASK_TIERS.get(task_type, 2)
        if tier >= 2 and len(text) < 20:
            return True

        # Model refused / error patterns
        refusal_markers = [
            "i cannot", "i can't", "i'm unable", "as an ai",
            "i don't have access", "error:", "exception:",
        ]
        lower = text.lower()
        if any(marker in lower for marker in refusal_markers):
            return True

        # Telemetry-based: if model has low reliability for this task type
        if self._telemetry is not None:
            reliability = self._telemetry.get_reliability(model, task_type)
            if reliability < 0.3 and reliability > 0.0:
                return True

        return False

    # ── Legacy static routing helpers ─────────────────────────────────

    def _pick_model_from_tier(self, tier: int) -> str:
        with self._lock:
            self.refresh_available_models()
            for candidate in self.TIERS.get(tier, []):
                resolved = self._resolve_available_variant(candidate)
                if resolved:
                    return resolved

            # Fallback to config or default if tier models are unavailable
            if tier == 1:
                return self._cfg("quick_model", self._cfg("chat_model", "llama3.2:1b"))
            elif tier == 3:
                return self._cfg("reasoning_model", self._cfg("chat_model", "deepseek-r1:8b"))
            else:
                return self._cfg("chat_model", "mistral:7b")

    # ── Availability ──────────────────────────────────────────────────

    def is_available(self, model_name: str) -> bool:
        with self._lock:
            self.refresh_available_models()
            model = str(model_name or "").strip()
            if not model or not self._available_ollama_models:
                return False
            if model in self._available_ollama_models:
                return True
            return self._resolve_available_variant(model) is not None

    def list_available(self) -> list[str]:
        with self._lock:
            self.refresh_available_models()
            return sorted(self._available_ollama_models)

    def refresh_available_models(
        self,
        base_url: str | None = None,
        *,
        force: bool = False,
        timeout_s: float = 3.0,
    ) -> list[str]:
        with self._lock:
            now = time.time()
            if (
                not force
                and self._available_ollama_models
                and (now - self._cache_time) < _MODEL_DISCOVERY_TTL_S
            ):
                return self.list_available_without_refresh()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # If cache is completely empty, do a synchronous fetch first
                # to prevent race conditions during early startup queries.
                if not self._initial_fetch_done:
                    try:
                        discovered = list_models_sync(
                            base_url=base_url or self._ollama_base_url(),
                            timeout_s=timeout_s,
                        )
                        self.set_available_models(discovered)
                        self._initial_fetch_done = True
                        return self.list_available_without_refresh()
                    except Exception:
                        self._initial_fetch_done = True
                        pass

                if not force:
                    self._cache_time = now - _MODEL_DISCOVERY_TTL_S + 5.0

                def _bg_update():
                    try:
                        discovered = list_models_sync(
                            base_url=base_url or self._ollama_base_url(),
                            timeout_s=timeout_s,
                        )
                        self.set_available_models(discovered)
                    except Exception:
                        pass

                loop.run_in_executor(None, _bg_update)
                return self.list_available_without_refresh()
            else:
                try:
                    discovered = list_models_sync(
                        base_url=base_url or self._ollama_base_url(),
                        timeout_s=timeout_s,
                    )
                    self.set_available_models(discovered)
                    self._initial_fetch_done = True
                except Exception:
                    self._initial_fetch_done = True
                    pass
                return self.list_available_without_refresh()

    def list_available_without_refresh(self) -> list[str]:
        with self._lock:
            return sorted(self._available_ollama_models)

    def set_available_models(self, models: Iterable[str]) -> None:
        with self._lock:
            self._available_ollama_models = {str(model) for model in models if str(model)}
            self._cache_time = time.time()

    # ── Internal helpers ──────────────────────────────────────────────

    def _cfg(self, key: str, default: str) -> str:
        if self.config is None:
            return default
        try:
            return str(self.config.get("models", key, fallback=default))
        except Exception:
            return default

    def _ollama_base_url(self) -> str:
        if self.config is not None:
            try:
                return str(self.config.get("ollama", "base_url", fallback=OLLAMA_BASE_URL))
            except Exception:
                pass
        return os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)

    def _resolve_available_variant(self, model_name: str) -> str | None:
        with self._lock:
            if not self._available_ollama_models:
                return None

            model = str(model_name or "").strip()
            if model in self._available_ollama_models:
                return model

            family = model.split(":", 1)[0]
            latest = f"{family}:latest"
            if latest in self._available_ollama_models:
                return latest

            for candidate in sorted(self._available_ollama_models):
                if candidate.split(":", 1)[0] == family:
                    return candidate
            return None


__all__ = ["ModelRouter"]
