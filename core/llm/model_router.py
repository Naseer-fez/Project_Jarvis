import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "planning": "deepseek-r1:8b",
    "chat": "mistral:7b",
    "vision": "llava",
    "synthesis": "deepseek-r1:8b",
    "embedding": "nomic-embed-text",
    "fallback": "mistral:7b",
}


class ModelRouter:
    def __init__(self, config: Optional[object] = None):
        self._models: dict[str, list[str]] = {}
        for key, val in DEFAULT_MODELS.items():
            self._models[key] = [m.strip() for m in val.split(",")]

        if config and config.has_section("models"):
            for key in DEFAULT_MODELS:
                opt = f"{key}_model" if key != "fallback" else "fallback_model"
                if config.has_option("models", opt):
                    raw = config.get("models", opt)
                    self._models[key] = [m.strip() for m in raw.split(",") if m.strip()]

        self._cache: dict = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 60.0
        self._available_ollama_models: set[str] = set()

    def route(self, task_type: str) -> str:
        """Returns the primary (first) model for the task type."""
        models = self._models.get(task_type, self._models["fallback"])
        return models[0] if models else "mistral:7b"

    def _refresh_cache_bg(self) -> None:
        try:
            import json
            import urllib.request
            from core.config.defaults import OLLAMA_BASE_URL

            with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as response:
                data = json.loads(response.read())
            available = {
                str(model.get("name", "")).strip()
                for model in data.get("models", [])
                if str(model.get("name", "")).strip()
            }
            flattened_models = [m for model_list in self._models.values() for m in model_list]
            self._available_ollama_models = available
            self._cache = {
                name: (self._resolve_ollama_model(name, available) is not None)
                for name in flattened_models
            }
            self._cache_time = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Model availability check failed: %s", exc)

    def _refresh_cache(self) -> None:
        if time.time() - self._cache_time < self._cache_ttl:
            return
        
        self._cache_time = time.time()
        
        if not self._cache:
            self._refresh_cache_bg()
        else:
            import threading
            threading.Thread(target=self._refresh_cache_bg, daemon=True).start()

    @staticmethod
    def _base_name(model_name: str) -> str:
        return model_name.split(":", 1)[0].strip()

    def _resolve_ollama_model(
        self,
        model_name: str,
        available: Optional[set[str]] = None,
    ) -> str | None:
        candidates = available if available is not None else self._available_ollama_models
        if not candidates:
            return None

        requested = model_name.strip()
        if not requested:
            return None
        if requested in candidates:
            return requested

        base = self._base_name(requested)
        latest = f"{base}:latest"

        if latest in candidates:
            return latest
        if base in candidates:
            return base

        family_matches = sorted(name for name in candidates if self._base_name(name) == base)
        if family_matches:
            return family_matches[0]

        return None

    def is_available(self, model_name: str) -> bool:
        if model_name.startswith("gemini-"):
            return bool(os.environ.get("GEMINI_API_KEY"))

        self._refresh_cache()
        return self._resolve_ollama_model(model_name) is not None

    def get_best_available(self, task_type: str) -> str:
        self._refresh_cache()
        candidates = self._models.get(task_type, self._models["fallback"])

        for candidate in candidates:
            if candidate.startswith("gemini-") and self.is_available(candidate):
                return candidate
            resolved = self._resolve_ollama_model(candidate)
            if resolved is not None:
                return resolved

        # If we got here, none of the specific candidates are available
        fallback_candidates = self._models["fallback"]
        for fallback in fallback_candidates:
            if fallback.startswith("gemini-") and self.is_available(fallback):
                logger.warning(
                    "No primary models available for task '%s'. Using fallback '%s'.",
                    task_type, fallback
                )
                return fallback
            resolved = self._resolve_ollama_model(fallback)
            if resolved is not None:
                logger.warning(
                    "No primary models available for task '%s'. Using fallback '%s'.",
                    task_type, resolved
                )
                return resolved


        # NOTHING is available — surface this loudly
        available = sorted(self._available_ollama_models)
        if available:
            logger.error(
                "No configured models available. Using first available Ollama model: '%s'. ",
                available[0]
            )
            return available[0]

        raise RuntimeError(
            f"No configured models available. Ollama is empty, and Gemini API key is missing.\n"
            f"Set GEMINI_API_KEY in .env, or run: ollama pull {fallback_candidates[0]}"
        )

    def list_available(self) -> dict:
        self._refresh_cache()
        return dict(self._cache)
