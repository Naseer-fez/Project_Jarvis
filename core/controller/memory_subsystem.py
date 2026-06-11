import asyncio
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

class MemorySubsystem:
    def __init__(self, memory: Any, profile: Any, synthesizer: Any, config: Any) -> None:
        self.memory = memory
        self.profile = profile
        self.synthesizer = synthesizer
        self.config = config
        
        self._synthesis_tasks: set[asyncio.Task] = set()
        self._conversation_buffer: List[str] = []
        self._runtime_loop: asyncio.AbstractEventLoop | None = None

    async def startup(self) -> None:
        self._runtime_loop = asyncio.get_running_loop()
        index_path = ""
        if hasattr(self.config, "get") and callable(self.config.get):
            # Assuming config is a configparser.ConfigParser
            try:
                index_path = self.config.get("memory", "index_path", fallback="")
            except Exception:
                pass

        self.memory_status = await self.memory.initialize(index_path=index_path)

    async def shutdown(self) -> None:
        # Cancel and wait for synthesis tasks
        for task in list(self._synthesis_tasks):
            task.cancel()
        if self._synthesis_tasks:
            try:
                await asyncio.gather(*self._synthesis_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning("Error gathering synthesis tasks during shutdown: %s", e, exc_info=True)

        if self.memory is not None:
            try:
                await self.memory.close()
            except Exception:
                logger.warning("Memory cleanup error during shutdown", exc_info=True)

    def update_profile(self, user_input: str, response: str) -> None:
        try:
            self.profile.update_from_conversation(user_input, response)
            self._conversation_buffer.append(f"User: {user_input}\nJarvis: {response}")

            if self.synthesizer.should_run(self.profile):
                self._schedule_synthesis(self._conversation_buffer[-20:])
                self._conversation_buffer.clear()
            elif len(self._conversation_buffer) > 50:
                self._conversation_buffer = self._conversation_buffer[-50:]
        except Exception as exc:
            logger.warning("Profile update/synthesis scheduling failed: %s", exc, exc_info=True)

    def _schedule_synthesis(self, conversations: List[str]) -> None:
        coro = self.synthesizer.synthesize(conversations, self.profile)
        
        def _track(task: asyncio.Task) -> None:
            self._synthesis_tasks.add(task)
            task.add_done_callback(self._synthesis_tasks.discard)

        try:
            task = asyncio.create_task(coro)
            _track(task)
            return
        except RuntimeError:
            pass

        if self._runtime_loop is not None and self._runtime_loop.is_running():
            def _create_and_track() -> None:
                t = asyncio.create_task(coro)
                _track(t)
            self._runtime_loop.call_soon_threadsafe(_create_and_track)
            return

        coro.close()
        logger.warning("No running loop available; skipped profile synthesis task.")
