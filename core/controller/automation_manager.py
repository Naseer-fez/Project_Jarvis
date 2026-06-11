import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

class AutomationManager:
    def __init__(
        self,
        config: Any,
        memory: Any,
        llm: Any,
        notifier: Any,
        desktop_observer: Any,
        container: Any,
        command_handler: Callable[[str], Any]
    ) -> None:
        self.config = config
        self.live_automation = None
        
        if hasattr(config, "has_section") and config.has_section("automation") and config.getboolean("automation", "enabled", fallback=False):
            try:
                from core.automation.live_automation import LiveAutomationEngine
                
                async def _handler(cmd: str) -> str:
                    return await command_handler(cmd)

                dag_executor = None
                if container is not None and hasattr(container, "has") and container.has("dag_executor"):
                    dag_executor = container.resolve("dag_executor")

                self.live_automation = LiveAutomationEngine(
                    config=config,
                    memory=memory,
                    llm=llm,
                    command_handler=_handler,
                    desktop_observer=desktop_observer,
                    notifier=notifier,
                    dag_executor=dag_executor,
                )
            except Exception as exc:
                logger.warning("Failed to initialize LiveAutomationEngine: %s", exc, exc_info=True)

    async def startup(self) -> None:
        if self.live_automation is not None:
            await self.live_automation.start()

    async def shutdown(self) -> None:
        if self.live_automation is not None:
            await self.live_automation.stop()
