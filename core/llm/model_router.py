"""Config-driven model router with lightweight Ollama tag compatibility."""

from __future__ import annotations

import time
from typing import Iterable


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

        if task in {"synthesis", "summary", "summarize"}:
            return self._cfg("summarize_model", "llama3.2:1b")
        if task in self.QUICK_TASKS:
            return self._cfg("quick_model", self._cfg("chat_model", "mistral:7b"))
        if task == "fallback":
            return self._cfg("fallback_model", self._cfg("chat_model", "mistral:7b"))
        if task in {"final_response", "chat", "general"}:
            return self._cfg("chat_model", "mistral:7b")

        return self._cfg("chat_model", "mistral:7b")

    def is_available(self, model_name: str) -> bool:
        model = str(model_name or "").strip()
        if not model or not self._available_ollama_models:
            return False
        if model in self._available_ollama_models:
            return True
        return self._resolve_available_variant(model) is not None

    def get_best_available(self, task_type: str) -> str:
        desired = self.route(task_type)
        available = self._resolve_available_variant(desired)
        return available or desired

    def _cfg(self, key: str, default: str) -> str:
        if self.config is None:
            return default
        try:
            return self.config.get("models", key, fallback=default)
        except Exception:
            return default

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
