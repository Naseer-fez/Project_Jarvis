"""
core/tools/vision.py — Vision (image analysis) tool.

Wraps Ollama's LLaVA model for image understanding.
Tests can patch _call_llava() to inject fake responses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


class VisionTool:
    """Analyze images using LLaVA or a test stub."""

    def __init__(self, config: Any) -> None:
        self._config = config
        self._model = self._get("vision_model", "llava")
        self._base_url = self._get("base_url", "http://localhost:11434")

    def _get(self, key: str, default: str) -> str:
        for section in ("ollama", "vision"):
            try:
                return str(self._config.get(section, key, fallback=default))
            except Exception:  # noqa: BLE001
                pass
        return default

    def analyze(self, image_path: str, prompt: str = "Describe this image.") -> str:
        """
        Analyze *image_path* using LLaVA and return a text description.

        Raises:
            FileNotFoundError: if the image file does not exist.
            ValueError: if the file extension is not a supported image format.
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix!r}. "
                f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        return self._call_llava(str(path), prompt)

    def _call_llava(self, image_path: str, prompt: str) -> str:
        """Call the LLaVA model. Override in tests."""
        try:
            import requests  # type: ignore[import]
            import base64

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return str(resp.json().get("response", ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLaVA call failed: %s", exc)
            return f"[Vision unavailable: {exc}]"


__all__ = ["VisionTool"]
