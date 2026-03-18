from __future__ import annotations

import asyncio
import configparser

from .controller import JarvisControllerV2


class Controller:
    """
    Compatibility controller for `main.py`.

    This adapter keeps the async lifecycle expected by `main.py`
    while using the stable Session-4 controller implementation.
    """

    def __init__(self, config: configparser.ConfigParser, voice: bool = False) -> None:
        del voice  # Voice loop is not implemented in the Session-4 adapter.
        self._config = config
        self._running = False

        db_path = config.get("memory", "sqlite_file", fallback="memory/memory.db")
        chroma_path = config.get("memory", "chroma_dir", fallback="data/chroma")
        planner_model = config.get("ollama", "planner_model", fallback="deepseek-r1:8b")
        embedding_model = config.get("memory", "embedding_model", fallback="all-MiniLM-L6-v2")

        self._impl = JarvisControllerV2(
            db_path=db_path,
            chroma_path=chroma_path,
            model_name=planner_model,
            embedding_model=embedding_model,
        )

    async def start(self) -> None:
        self._impl.initialize()
        self._running = True

    async def run_cli(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                user_input = await loop.run_in_executor(None, input, "You> ")
            except (EOFError, KeyboardInterrupt):
                self._running = False
                break

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                self._running = False
                break

            response = await loop.run_in_executor(None, self._impl.process, text)
            print(f"Jarvis> {response}")

    async def shutdown(self) -> None:
        self._running = False
