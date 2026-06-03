from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Iterable, Any

from core.config.defaults import OLLAMA_BASE_URL
from core.llm.ollama_client import list_models_sync

_MODEL_DISCOVERY_TTL_S = 30.0

class ModelRouter:
    """Intelligent multi-stage model router for Jarvis."""

    TIERS = {
        1: ["llama3.2:1b", "qwen2.5:0.5b", "qwen2.5:1.5b", "gemma2:2b"],
        2: ["mistral:7b", "llama3:8b", "qwen2.5:7b"],
        3: ["deepseek-r1:8b", "deepseek-r1:14b", "llama3.3:70b"]
    }

    TASK_TIERS = {
        "intent": 1,
        "intent_classification": 1,
        "reflex": 1,
        "web_search_summary": 1,
        "tool_parameter_extraction": 1,
        "context_title_generation": 1,
        "synthesis": 1,
        "summarize": 1,
        "memory_summarization": 1,

        "chat": 2,
        "general": 2,
        "planning": 2,
        "plan": 2,
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
        self._cache_time = 0.0
        self._lock = threading.RLock()
        
        # Override TIERS from config if provided
        if self.config and self.config.has_section("routing"):
            for tier_num in (1, 2, 3):
                val = self.config.get("routing", f"tier{tier_num}", fallback="")
                if val:
                    self.TIERS[tier_num] = [m.strip() for m in val.split(",") if m.strip()]

    def route(self, task_type: str) -> str:
        """Determines the target tier for a task and returns the best available model."""
        task = str(task_type or "chat").strip().lower()
        
        # Vision is a special case
        if task == "vision":
            return self._cfg("vision_model", "llava:latest")

        target_tier = self.TASK_TIERS.get(task, 2)
        
        return self._pick_model_from_tier(target_tier)

    def escalate(self, current_model: str) -> str:
        """Upgrades the model to a higher tier if possible."""
        current_tier = 1
        for t, models in self.TIERS.items():
            if current_model in models or any(current_model.startswith(m.split(":")[0]) for m in models):
                current_tier = t
                break
        
        next_tier = min(3, current_tier + 1)
        return self._pick_model_from_tier(next_tier)

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

    def is_available(self, model_name: str) -> bool:
        with self._lock:
            self.refresh_available_models()
            model = str(model_name or "").strip()
            if not model or not self._available_ollama_models:
                return False
            if model in self._available_ollama_models:
                return True
            return self._resolve_available_variant(model) is not None

    def pick_model(self, task_type: str) -> str:
        """Public entry point — returns the best model name for a task type."""
        return self.get_best_available(task_type)

    def get_best_available(self, task_type: str) -> str:
        return self.route(task_type)

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
                except Exception:
                    pass
                return self.list_available_without_refresh()

    def list_available_without_refresh(self) -> list[str]:
        with self._lock:
            return sorted(self._available_ollama_models)

    def _cfg(self, key: str, default: str) -> str:
        if self.config is None:
            return default
        try:
            return self.config.get("models", key, fallback=default)
        except Exception:
            return default

    def _ollama_base_url(self) -> str:
        if self.config is not None:
            try:
                return self.config.get("ollama", "base_url", fallback=OLLAMA_BASE_URL)
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

    def set_available_models(self, models: Iterable[str]) -> None:
        with self._lock:
            self._available_ollama_models = {str(model) for model in models if str(model)}
            self._cache_time = time.time()


__all__ = ["ModelRouter"]
