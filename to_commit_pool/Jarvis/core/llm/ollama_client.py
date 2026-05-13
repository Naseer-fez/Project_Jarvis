"""Pure async HTTP client for local Ollama inference.

Talks to Ollama's /api/generate endpoint.  No cloud fallback,
no memory injection, no profile — just HTTP in, text out.
"""

from __future__ import annotations

import asyncio
import logging
import re

import aiohttp

from core.config.defaults import OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-r1:8b"
TIMEOUT_S = 120


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by DeepSeek R1."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


class OllamaClient:
    """Lightweight async client for a local Ollama instance.

    Usage::

        client = OllamaClient()
        reply = await client.complete("say hello", model="mistral:7b")
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        *,
        model: str = DEFAULT_MODEL,
        keep_think: bool = False,
    ) -> str:
        """Send a prompt to Ollama and return the response text.

        Retries up to 3 times on transient connection errors.
        Raises on timeout, connection refused, or empty response.
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9},
        }
        if system:
            payload["system"] = system

        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_S),
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                f"Ollama HTTP {response.status} on attempt {attempt + 1}"
                            )
                        data = await response.json()
                        raw = str(data.get("response", ""))
                        if not raw.strip():
                            raise RuntimeError("Ollama returned empty response")
                        if not keep_think:
                            raw = _strip_think(raw)
                        return raw

            except asyncio.TimeoutError as exc:
                logger.error("Ollama timeout after %ss (attempt %d)", TIMEOUT_S, attempt + 1)
                last_exc = exc
                break  # timeouts are not transient — don't retry

            except aiohttp.ClientError as exc:
                logger.warning("Ollama connection error (attempt %d): %s", attempt + 1, exc)
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))

            except RuntimeError as exc:
                logger.error("Ollama error: %s", exc)
                last_exc = exc
                break

            except Exception as exc:  # noqa: BLE001
                logger.error("Ollama unexpected failure: %s", exc)
                last_exc = exc
                break

        raise last_exc or RuntimeError("Ollama request failed after 3 attempts")

    async def is_running(self) -> bool:
        """Quick health check — GET the Ollama root endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as response:
                    return response.status == 200
        except Exception:
            return False


__all__ = ["OllamaClient"]
