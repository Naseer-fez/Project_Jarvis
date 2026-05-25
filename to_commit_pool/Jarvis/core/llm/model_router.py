"""Config-driven model router with lightweight Ollama tag compatibility."""

from __future__ import annotations

import os
import time
from typing import Iterable

from core.config.defaults import OLLAMA_BASE_URL
from core.llm.ollama_client import list_models_sync


_MODEL_DISCOVERY_TTL_S = 30.0


class ModelRouter:
    QUICK_TASKS = {
        "web_search_summary",
        "tool_parameter_extraction",
        "context_title_generation",
    }

    def __init__(self, config=None) -> None:
        self.config = config
        self._available_ollama_models: set[str] = set()
        self._cache_time = 0.0

    def route(self, task_type: str) -> str:
        task = str(task_type or "chat").strip().lower()

        if task in {"intent", "intent_classification"}:
            return self._cfg(
                "intent_model",
                self._cfg("quick_model", self._cfg("chat_model", "mistral:7b")),
            )
        if task in {"planning", "plan"}:
            return self._cfg("plan_model", self._cfg("chat_model", "mistral:7b"))
        if task in {"tool_selection", "tool_picker"}:
            return self._cfg("tool_picker_model", self._cfg("chat_model", "mistral:7b"))
        if task in {"synthesis", "summary", "summarize"}:
            return self._cfg("summarize_model", "llama3.2:1b")
        if task in {"memory_summarization"}:
            return self._cfg("summarize_model", self._cfg("chat_model", "mistral:7b"))
        if task == "vision":
            return self._cfg(
                "vision_model",
                self._cfg("chat_model", self._cfg("fallback_model", "llava:latest")),
            )
        if task in self.QUICK_TASKS:
            return self._cfg("quick_model", self._cfg("chat_model", "mistral:7b"))
        if task == "fallback":
            return self._cfg("fallback_model", self._cfg("chat_model", "mistral:7b"))
        if task in {"final_response", "chat", "general"}:
            return self._cfg("chat_model", "mistral:7b")

        return self._cfg("chat_model", "mistral:7b")

    def is_available(self, model_name: str) -> bool:
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
        self.refresh_available_models()
        desired = self.route(task_type)
        available = self._resolve_available_variant(desired)
        return available or desired

    def list_available(self) -> list[str]:
        self.refresh_available_models()
        return sorted(self._available_ollama_models)

    def refresh_available_models(
        self,
        base_url: str | None = None,
        *,
        force: bool = False,
        timeout_s: float = 3.0,
    ) -> list[str]:
        if (
            not force
            and self._available_ollama_models
            and (time.time() - self._cache_time) < _MODEL_DISCOVERY_TTL_S
        ):
            return self.list_available_without_refresh()

        try:
            discovered = list_models_sync(
                base_url=base_url or self._ollama_base_url(),
                timeout_s=timeout_s,
            )
        except Exception:
            return self.list_available_without_refresh()

        self.set_available_models(discovered)
        return self.list_available_without_refresh()

    def list_available_without_refresh(self) -> list[str]:
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
        self._available_ollama_models = {str(model) for model in models if str(model)}
        self._cache_time = time.time()


__all__ = ["ModelRouter"]
