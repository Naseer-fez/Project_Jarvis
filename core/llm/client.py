"""
core/llm/client.py
═══════════════════
Async LLM client for Jarvis V1.

Model  : deepseek-r1:8b (via Ollama)
Rules  :
  - Offline only — localhost:11434
  - Strips <think>...</think> reasoning blocks from output
  - Never raises — returns empty string on any failure
  - All calls logged
  - Configurable system prompt per call
"""

import asyncio
import re
import json
from datetime import datetime, timezone
from core.logger import get_logger

logger = get_logger("llm")

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-r1:8b"
TIMEOUT_S     = 120


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks DeepSeek R1 adds before answering."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences if present."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


class LLMClientV2:
    """
    Async wrapper around Ollama's /api/generate endpoint.

    Usage:
        llm = LLMClientV2()
        response = await llm.complete("What is 2+2?")
        json_obj = await llm.complete_json("Return a JSON object with key 'answer'")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_URL,
        timeout: int = TIMEOUT_S,
    ):
        self.model   = model
        self.base_url = base_url
        self.timeout  = timeout

    async def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """
        Send a prompt to the LLM and return the text response.
        Returns "" on any failure — never raises.
        """
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return ""

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            }
        }
        if system:
            payload["system"] = system

        logger.debug(f"LLM call | model={self.model} | prompt_len={len(prompt)}")
        start = datetime.now(timezone.utc)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Ollama HTTP {resp.status}: {body[:200]}")
                        return ""

                    data = await resp.json()
                    raw  = data.get("response", "")

        except aiohttp.ClientConnectorError:
            logger.error(
                "Cannot connect to Ollama. "
                "Make sure it is running: ollama serve"
            )
            return ""
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {self.timeout}s")
            return ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        cleaned = _strip_think_blocks(raw)
        logger.debug(f"LLM response | len={len(cleaned)} | elapsed={elapsed:.1f}s")
        return cleaned

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> dict | None:
        """
        Like complete(), but parses and returns a JSON dict.
        Returns None if output is not valid JSON.
        """
        raw = await self.complete(prompt, system=system, temperature=temperature)
        if not raw:
            return None

        cleaned = _strip_code_fences(raw)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}\nRaw: {cleaned[:300]}")
            return None

    async def is_ollama_running(self) -> bool:
        """Quick health check — returns True if Ollama is reachable."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:11434",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False