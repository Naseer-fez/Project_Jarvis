from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

class CloudLLMClient:
    """
    Unified interface for cloud LLM providers.
    Priority: Groq (fastest, cheapest) → OpenAI → Anthropic → fail loudly.
    """

    PROVIDERS = ["groq", "openai", "anthropic"]

    def __init__(self) -> None:
        self._available: list[str] = []
        for provider in self.PROVIDERS:
            if self._check_provider(provider):
                self._available.append(provider)
        if not self._available:
            logger.warning("No cloud LLM providers configured. Cloud fallback disabled.")

    def _check_provider(self, name: str) -> bool:
        keys = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        return bool(os.environ.get(keys.get(name, "")))

    async def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        for provider in self._available:
            try:
                result = await self._call(provider, prompt, system, temperature)
                if result:
                    logger.info("Cloud LLM response from '%s'", provider)
                    return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cloud provider '%s' failed: %s", provider, exc)
        raise RuntimeError("All cloud LLM providers failed or are unconfigured.")

    async def _call(self, provider: str, prompt: str, system: str, temperature: float) -> str:
        if provider == "groq":
            return await self._call_groq(prompt, system, temperature)
        if provider == "openai":
            return await self._call_openai(prompt, system, temperature)
        if provider == "anthropic":
            return await self._call_anthropic(prompt, system, temperature)
        return ""

    async def _call_groq(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",  # fastest Groq model as of 2025
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2048,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
                if "choices" not in data or not data["choices"]:
                    logger.debug("Groq response missing choices: %s", data)
                    return ""
                return str(data["choices"][0]["message"]["content"])

    async def _call_openai(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",  # cheapest capable model
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
                if "choices" not in data or not data["choices"]:
                    logger.debug("OpenAI response missing choices: %s", data)
                    return ""
                return str(data["choices"][0]["message"]["content"])

    async def _call_anthropic(self, prompt: str, system: str, temperature: float) -> str:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307", # Use claude-3-haiku because haiku 3.5 hasn't historically behaved properly in old prompt suites without specific tweaks
                    "max_tokens": 2048,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
                if "content" not in data or not data["content"]:
                    logger.debug("Anthropic response missing content: %s", data)
                    return ""
                return str(data["content"][0]["text"])

__all__ = ["CloudLLMClient"]
