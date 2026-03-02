"""JarvisControllerV2: memory + LLM orchestration with CLI/voice runtime modes."""

from __future__ import annotations

import asyncio
import configparser
import uuid
from typing import Any

from core.llm.llm_v2 import LLMClientV2
from core.memory.hybrid_memory import HybridMemory


class JarvisControllerV2:
    def __init__(
        self,
        config: configparser.ConfigParser | None = None,
        voice: bool = False,
        db_path: str = "memory/memory.db",
        chroma_path: str = "data/chroma",
        model_name: str = "deepseek-r1:8b",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.config = config if isinstance(config, configparser.ConfigParser) else configparser.ConfigParser()
        self.voice_enabled = bool(voice)

        if isinstance(config, configparser.ConfigParser):
            db_path = config.get("memory", "db_path", fallback=config.get("memory", "sqlite_file", fallback=db_path))
            chroma_path = config.get("memory", "chroma_path", fallback=config.get("memory", "chroma_dir", fallback=chroma_path))
            model_name = config.get("llm", "model", fallback=config.get("ollama", "planner_model", fallback=model_name))
            embedding_model = config.get("memory", "embedding_model", fallback=embedding_model)

        self.session_id = uuid.uuid4().hex[:8]
        self.memory = HybridMemory(db_path, chroma_path=chroma_path, model_name=embedding_model)
        self.llm = LLMClientV2(model_name=model_name)
        self.exchanges = 0

        self.voice_loop = None

    def initialize(self) -> dict[str, Any]:
        index_path = ""
        if isinstance(self.config, configparser.ConfigParser):
            index_path = self.config.get("memory", "index_path", fallback="")

        memory_status = self.memory.initialize(index_path=index_path)
        return {
            "session_id": self.session_id,
            "memory_mode": memory_status.get("mode"),
            "memory_init": memory_status,
        }

    def process(self, user_input: str) -> str:
        self.exchanges += 1
        text = (user_input or "").strip()
        lowered = text.lower()

        if lowered == "status":
            return f"Session: {self.session_id} | Memory Mode: {self.memory.mode}"
        if lowered == "help":
            return "Commands: status, help, exit, remember <fact>, what's <query>"

        if lowered.startswith("remember i like "):
            value = text[16:].strip()
            if value:
                self.memory.store_preference(f"likes_{value[:12]}", value)
                return f"I will remember you like {value}."

        if lowered.startswith("my name is "):
            value = text[11:].strip()
            if value:
                self.memory.store_preference("name", value)
                return f"I will remember your name is {value}."

        if lowered.startswith("i prefer "):
            value = text[9:].strip()
            if value:
                self.memory.store_preference(f"prefer_{value[:12]}", value)
                return f"I will remember you prefer {value}."

        if lowered.startswith("i work in "):
            value = text[10:].strip()
            if value:
                self.memory.store_preference("work", value)
                return f"I will remember you work in {value}."

        context = self.memory.build_context_block(text)
        response = self.llm.chat(text, system_context=context)

        if response == "LLM Offline.":
            prefs = self.memory.recall_preferences(text, top_k=1)
            if prefs and prefs[0].get("value"):
                return f"Offline fallback from memory: {prefs[0]['value']}"
            return "I don't know while offline."

        self.memory.store_conversation(text, response, self.session_id)
        return response

    def session_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "exchanges": self.exchanges,
        }

    async def start(self) -> None:
        self.initialize()

    async def run_cli(self) -> None:
        """Run either standard CLI or voice loop based on --voice/config."""
        if self.voice_enabled:
            from core.voice.voice_loop import VoiceLoop

            self.voice_loop = VoiceLoop(controller=self, config=self.config)
            await self.voice_loop.run()
            return

        print(f"Jarvis V2 ready (session {self.session_id}). Type 'exit' to quit.")
        loop = asyncio.get_running_loop()

        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "You: ")
            except EOFError:
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break

            response = await loop.run_in_executor(None, self.process, text)
            print(f"Jarvis: {response}")

    async def shutdown(self) -> None:
        if self.voice_loop is not None:
            await self.voice_loop.stop()


# Backward-compatible alias used by main.py
Controller = JarvisControllerV2

__all__ = ["JarvisControllerV2", "Controller"]
