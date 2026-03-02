import logging
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
        self._models = dict(DEFAULT_MODELS)
        if config and config.has_section("models"):
            for key in DEFAULT_MODELS:
                opt = f"{key}_model" if key != "fallback" else "fallback_model"
                if config.has_option("models", opt):
                    self._models[key] = config.get("models", opt)
        self._cache: dict = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = 60.0

    def route(self, task_type: str) -> str:
        return self._models.get(task_type, self._models["fallback"])

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
        self._refresh_cache()
        return self._cache.get(model_name, False)

    def get_best_available(self, task_type: str) -> str:
        preferred = self.route(task_type)
        if self.is_available(preferred):
            return preferred

        fallback = self._models["fallback"]
        if self.is_available(fallback):
            logger.warning("Model %s unavailable, using fallback %s", preferred, fallback)
            return fallback

        # Last resort: return fallback even if not confirmed available.
        return fallback

    def list_available(self) -> dict:
        self._refresh_cache()
        return dict(self._cache)
