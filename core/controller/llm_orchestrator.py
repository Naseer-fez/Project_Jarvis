import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self, llm_dispatcher: Any) -> None:
        self.llm_dispatcher = llm_dispatcher
        self._inflight_llm_calls = 0

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        # Wait for inflight calls to complete
        while self._inflight_llm_calls > 0:
            await asyncio.sleep(0.1)

    async def dispatch(self, text: str, classification: Dict[str, Any], session_id: str, trace_id: str) -> str:
        self._inflight_llm_calls += 1
        try:
            return await self.llm_dispatcher.dispatch(text, classification, session_id, trace_id)
        finally:
            self._inflight_llm_calls -= 1
