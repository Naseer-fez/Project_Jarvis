"""
core/tools/vision.py — Passive vision via LLaVA.

RULES (enforced by design):
  - Read-only: only produces text descriptions, never triggers actions
  - No side effects beyond writing the description to memory
  - Image path must exist and be a supported format before calling
  - Never called automatically — must be explicitly invoked by user or planner

Authority level: L0_OBSERVE
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

_VISION_SYSTEM = (
    "You are a passive visual observer. Describe what you see in the image accurately "
    "and concisely. Do not suggest any actions. Do not interpret intent. "
    "Do not trigger or recommend any computer operations."
)


class VisionTool:
    def __init__(self, config) -> None:
        self._base_url = config.get("ollama", "base_url", fallback="http://localhost:11434")
        self._model    = config.get("ollama", "vision_model", fallback="llava")
        self._timeout  = int(config.get("ollama", "request_timeout_s", fallback="120"))

    def analyze(self, image_path: str, prompt: str = "Describe this image.") -> str:
        """
        Analyze an image. Returns a text description.
        Raises FileNotFoundError or ValueError for bad inputs.
        Never triggers actions.
        """
        path = Path(image_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(_SUPPORTED_EXTENSIONS)}"
            )

        image_b64 = self._encode_image(path)
        return self._call_llava(image_b64, prompt)

    def _encode_image(self, path: Path) -> str:
        with path.open("rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _call_llava(self, image_b64: str, prompt: str) -> str:
        url = f"{self._base_url}/api/generate"
        payload = json.dumps({
            "model":  self._model,
            "prompt": prompt,
            "system": _VISION_SYSTEM,
            "images": [image_b64],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        return data.get("response", "(no response from vision model)")

    def ping(self) -> bool:
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False
