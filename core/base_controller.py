import abc
import asyncio
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

class BaseController(abc.ABC):
    """
    Abstract Base Class for Controllers.
    Enforces startup and shutdown methods for subclasses, and provides
    implementations using asyncio.TaskGroup to synchronize the 
    startup and shutdown of child modules (injectable subsystems).
    """

    def __init__(self) -> None:
        self._subsystems: List[Any] = []

    def register_subsystem(self, subsystem: Any) -> None:
        """Registers a child module/subsystem to be synchronized."""
        self._subsystems.append(subsystem)

    @abc.abstractmethod
    async def startup(self) -> None:
        """
        Starts up the controller.
        Subclasses MUST override this method.
        To synchronize the startup of registered subsystems, call `await super().startup()`.
        """
        if self._subsystems:
            logger.info(f"{self.__class__.__name__}: Starting {len(self._subsystems)} subsystems concurrently...")
            async with asyncio.TaskGroup() as tg:
                for subsystem in self._subsystems:
                    if hasattr(subsystem, "startup") and callable(subsystem.startup):
                        tg.create_task(subsystem.startup())
            logger.info(f"{self.__class__.__name__}: Subsystem startup complete.")

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """
        Shuts down the controller.
        Subclasses MUST override this method.
        To synchronize the shutdown of registered subsystems, call `await super().shutdown()`.
        """
        if self._subsystems:
            logger.info(f"{self.__class__.__name__}: Shutting down {len(self._subsystems)} subsystems concurrently...")
            async with asyncio.TaskGroup() as tg:
                for subsystem in self._subsystems:
                    if hasattr(subsystem, "shutdown") and callable(subsystem.shutdown):
                        tg.create_task(subsystem.shutdown())
            logger.info(f"{self.__class__.__name__}: Subsystem shutdown complete.")
