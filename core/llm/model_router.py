import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "intent": "qwen2.5:0.5b",
    "summarize": "llama3.2:1b",
    "chat": "mistral:7b",
    "tool_picker": "mistral:7b",
    "plan": "deepseek-r1:8b",
    "fallback": "gemini-2.5-flash",
    "vision": "llava",
}

class ModelRouter:
    TASK_MAP = {
        "intent": ["intent", "chat", "fallback"],
        "intent_classification": ["intent", "chat", "fallback"],
        "summarize": ["summarize", "chat", "fallback"],
        "memory_summarization": ["summarize", "chat", "fallback"],
        "synthesis": ["summarize", "chat", "fallback"],
        "tool_picker": ["tool_picker", "plan", "fallback"],
        "tool_selection": ["tool_picker", "plan", "fallback"],
        "plan": ["plan", "fallback"],
        "planning": ["plan", "fallback"],
        "chat": ["chat", "plan", "fallback"],
        "vision": ["vision", "fallback"],
        "fallback": ["fallback"],
    }

    def __init__(self, config: Optional[object] = None):
        self._models: dict[str, list[str]] = {}
        self.config = config
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

        self._base_url = "http://localhost:11434"
        if config and config.has_section("ollama"):
            self._base_url = config.get("ollama", "base_url", fallback=self._base_url)

    def _resolve_task_candidates(self, task_type: str | None) -> list[str]:
        normalized = str(task_type or "chat").strip().lower() or "chat"
        return self.TASK_MAP.get(normalized, ["chat", "fallback"])

    def route(self, task_type: str = "chat") -> str:
        """Returns the primary (first) model for the task type."""
        candidates = self._resolve_task_candidates(task_type)
        models = []
        for c in candidates:
            # We try to use the configured model, and then fall back to defaults
            cfg_models = self._models.get(c, self._models.get("fallback", ["mistral:7b"]))
            if cfg_models:
                models.extend(cfg_models)

        return models[0] if models else "mistral:7b"

    def _refresh_cache_bg(self) -> None:
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=3) as response:
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

    def get_best_available(self, task_type: str = "chat") -> str | None:
        self._refresh_cache()
        candidates = self._resolve_task_candidates(task_type)

        for candidate in candidates:
            # Check configured string
            model_list = self._models.get(candidate)
            if not model_list:
                continue
            for model_name in model_list:
                if model_name.startswith("gemini-") and self.is_available(model_name):
                    return model_name
                resolved = self._resolve_ollama_model(model_name)
                if resolved is not None:
                    return resolved

        # If we got here, none of the specific candidates are available
        fallback_candidates = self._models.get("fallback", [])
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

        fallback_hint = fallback_candidates[0] if fallback_candidates else "a local Ollama model"
        raise RuntimeError(
            f"No configured models available. Ollama is empty, and Gemini API key is missing.\n"
            f"Set GEMINI_API_KEY in .env, or run: ollama pull {fallback_hint}"
        )

    def list_available(self) -> dict:
        self._refresh_cache()
        return dict(self._cache)
