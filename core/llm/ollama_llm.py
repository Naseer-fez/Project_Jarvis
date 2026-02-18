"""
OllamaLLM — thin async wrapper around the Ollama /api/chat endpoint.
Manages conversation history for multi-turn interactions.
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger("Jarvis.OllamaLLM")

JARVIS_SYSTEM_PROMPT = """You are Jarvis, an intelligent, efficient, and helpful AI assistant.
You run locally on the user's machine. You are honest about your capabilities and limitations.
You can reason, plan, and use tools. You are concise but complete in your responses.
When asked to do something, think step by step. When uncertain, say so."""


class OllamaLLM:
    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        system_prompt: str = JARVIS_SYSTEM_PROMPT,
    ):
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self._history: list[dict] = []

    def reset_history(self):
        self._history.clear()

    async def chat(self, user_message: str, inject_context: str = "") -> Optional[str]:
        """
        Send a message and get a response. Maintains history for multi-turn.
        inject_context: optional text prepended to the system context (e.g., memory summary).
        """
        system = self.system_prompt
        if inject_context:
            system = f"{system}\n\n{inject_context}"

        messages = [{"role": "system", "content": system}] + self._history + [
            {"role": "user", "content": user_message}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                reply = data["message"]["content"].strip()

                # Update history
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": reply})

                # Keep history bounded (last 20 exchanges = 40 messages)
                if len(self._history) > 40:
                    self._history = self._history[-40:]

                return reply

        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Run: ollama serve")
            return "⚠️ Cannot reach the local Ollama server. Please run `ollama serve` and try again."
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"⚠️ LLM error: {e}"

    async def check_availability(self) -> bool:
        """Ping Ollama to check if it's running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                tags = resp.json()
                models = [m["name"].split(":")[0] for m in tags.get("models", [])]
                if self.model in models or any(self.model in m for m in models):
                    logger.info(f"Ollama OK — model '{self.model}' available.")
                    return True
                else:
                    logger.warning(
                        f"Model '{self.model}' not found. Available: {models}. "
                        f"Run: ollama pull {self.model}"
                    )
                    return False
        except Exception:
            return False
