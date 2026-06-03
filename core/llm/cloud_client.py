from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class CloudLLMClient:
    """Best-effort cloud fallback across a small provider chain, with Tier-aware routing."""

    PROVIDERS = ["gemini", "groq", "openai", "anthropic"]

    # Tiered models for each provider
    MODELS = {
        "gemini": {
            1: "gemini-1.5-flash",
            2: "gemini-1.5-pro",
            3: "gemini-2.0-pro-exp-02-05",
        },
        "groq": {
            1: "llama-3.1-8b-instant",
            2: "llama-3.3-70b-versatile",
            3: "deepseek-r1-distill-llama-70b",
        },
        "openai": {
            1: "gpt-4o-mini",
            2: "gpt-4o",
            3: "o3-mini",
        },
        "anthropic": {
            1: "claude-3-haiku-20240307",
            2: "claude-3-5-sonnet-20241022",
            3: "claude-3-5-sonnet-20241022", # Anthropic doesn't have a distinct tier 3 right now, sonnet is very capable
        }
    }

    def __init__(self) -> None:
        provider_keys = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        self._available = [
            provider
            for provider in self.PROVIDERS
            if os.environ.get(provider_keys[provider])
        ]
        if not self._available:
            logger.warning("No cloud LLM providers configured. Cloud fallback disabled.")

    async def complete(self, prompt: str, system: str = "", temperature: float = 0.1, tier: int = 2) -> str:
        for provider in self._available:
            try:
                model = self.MODELS[provider].get(tier, self.MODELS[provider][2])
                response = await self._call(provider, prompt, system, temperature, model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cloud provider '%s' failed: %s", provider, exc)
                continue

            if response:
                logger.info("Cloud LLM response from '%s' using model '%s'", provider, model)
                return response

        raise RuntimeError(f"All cloud LLM providers failed for tier {tier}.")

    async def _call(self, provider: str, prompt: str, system: str, temperature: float, model: str) -> str:
        if provider == "gemini":
            return await self._call_gemini(prompt, system, temperature, model)
        if provider == "groq":
            return await self._call_groq(prompt, system, temperature, model)
        if provider == "openai":
            return await self._call_openai(prompt, system, temperature, model)
        if provider == "anthropic":
            return await self._call_anthropic(prompt, system, temperature, model)
        return ""

    async def _call_groq(self, prompt: str, system: str, temperature: float, model: str) -> str:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 2048,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        if not data.get("choices"):
            logger.debug("Groq response missing choices: %s", data)
            return ""
        return str(data["choices"][0]["message"]["content"])

    async def _call_openai(self, prompt: str, system: str, temperature: float, model: str) -> str:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        if not data.get("choices"):
            logger.debug("OpenAI response missing choices: %s", data)
            return ""
        return str(data["choices"][0]["message"]["content"])

    async def _call_anthropic(self, prompt: str, system: str, temperature: float, model: str) -> str:
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
                    "model": model,
                    "max_tokens": 2048,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        if not data.get("content"):
            logger.debug("Anthropic response missing content: %s", data)
            return ""
        return str(data["content"][0]["text"])

    async def _call_gemini(self, prompt: str, system: str, temperature: float, model: str) -> str:
        import aiohttp

        api_key = os.environ["GEMINI_API_KEY"]
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
        try:
            return str(data["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError):
            logger.debug("Gemini response missing content: %s", data)
            return ""


__all__ = ["CloudLLMClient"]
