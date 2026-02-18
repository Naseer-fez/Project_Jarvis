"""
core/llm.py
───────────
LLM Client V3 (Session 7).
Updated to support Identity-Aware System Prompts.
"""

import json
import logging
import requests
from typing import Optional, Generator

from memory.hybrid_memory import HybridMemory
from core.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL  = "http://localhost:11434"
DEFAULT_MODEL    = "deepseek-r1:8b"

SYSTEM_PROMPT_TEMPLATE = """\
You are Jarvis, a personal AI assistant.

{profile_block}

RULES:
- Always respond in English.
- Do not use emojis unless the user profile suggests a casual style.
- Be concise by default, unless the profile requests detail.
- If you do not know something, say so honestly.

{memory_block}
"""

class LLMClientV2:
    def __init__(
        self,
        hybrid_memory: Optional[HybridMemory] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.hybrid_memory  = hybrid_memory
        self.compressor     = ContextCompressor()
        self.model          = model
        self.base_url       = OLLAMA_BASE_URL
        self._chat_url      = f"{self.base_url}/api/chat"

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=2)
            return True
        except:
            return False

    def _build_system_prompt(self, query: str, profile_summary: str = "") -> str:
        memory_block = ""
        if self.hybrid_memory:
            try:
                # Session 7: Profile can influence recall? (Future optimization)
                recall = self.hybrid_memory.recall_all(query, top_k=5)
                memory_block = self.compressor.compress(query, recall)
            except Exception as e:
                logger.warning(f"Memory recall failed: {e}")

        return SYSTEM_PROMPT_TEMPLATE.format(
            profile_block=profile_summary,
            memory_block=memory_block if memory_block else "(No specific memory context)"
        )

    def chat(self, messages: list[dict], query_for_memory: Optional[str] = None, profile_summary: str = "") -> str:
        if not self.is_available():
            return "[Error: Ollama is offline]"
            
        sys_prompt = self._build_system_prompt(
            query_for_memory or messages[-1]["content"],
            profile_summary
        )
        
        payload = {
            "model": self.model,
            "stream": False,
            "system": sys_prompt,
            "messages": messages
        }
        
        try:
            r = requests.post(self._chat_url, json=payload, timeout=60)
            return r.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"[Error: {e}]"

    def chat_stream(
        self, 
        messages: list[dict], 
        query_for_memory: Optional[str] = None, 
        profile_summary: str = ""
    ) -> Generator[str, None, None]:
        if not self.is_available():
            yield "[Error: Ollama is offline]"
            return

        sys_prompt = self._build_system_prompt(
            query_for_memory or messages[-1]["content"], 
            profile_summary
        )
        
        payload = {
            "model": self.model,
            "stream": True,
            "system": sys_prompt,
            "messages": messages
        }
        
        try:
            with requests.post(self._chat_url, json=payload, stream=True, timeout=60) as r:
                for line in r.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content: yield content
                        except: pass
        except Exception as e:
            yield f"[Stream Error: {e}]"