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

    def route(self, task_type: str) -> str:
        """Returns the primary (first) model for the task type."""
        models = self._models.get(task_type, self._models["fallback"])
        return models[0] if models else "mistral:7b"

    def _refresh_cache(self) -> None:
        if time.time() - self._cache_time < self._cache_ttl:
            return
        try:
            import json
            import urllib.request

            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
                data = json.loads(response.read())
            available = {model["name"] for model in data.get("models", [])}
            self._cache = {name: (name in available) for name in self._models.values()}
            self._cache_time = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Model availability check failed: %s", exc)
            # Keep stale cache - do not clear it.

    def is_available(self, model_name: str) -> bool:
        if model_name.startswith("gemini-"):
            return bool(os.environ.get("GEMINI_API_KEY"))

        self._refresh_cache()
        for cached_model in self._cache:
            if cached_model == model_name or cached_model.startswith(model_name + ":"):
                return True
        return False

    def get_best_available(self, task_type: str) -> str:
        candidates = self._models.get(task_type, self._models["fallback"])
        
        for candidate in candidates:
            if self.is_available(candidate):
                return candidate
        if self.is_available(fallback):
            logger.warning(
                "Preferred model '%s' unavailable for task '%s'. Using fallback '%s'.",
                preferred, task_type, fallback
            )
        # If we got here, none of the specific candidates are available
        fallback_candidates = self._models["fallback"]
        for fallback in fallback_candidates:
            if self.is_available(fallback):
                logger.warning(
                    "No primary models available for task '%s'. Using fallback '%s'.",
                    task_type, fallback
                )
                return fallback

        # NOTHING is available — surface this loudly
        available = [m for m, ok in self.list_available().items() if ok]
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
