"""
core/kernel/event_bus.py
────────────────────────
Asynchronous event bus supporting non-blocking pub/sub.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

logger = logging.getLogger("Jarvis.Kernel.EventBus")


class EventBus:
    """Publish/Subscribe Event Bus allowing loose coupling between modules."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register a callback for a specific event type."""
        event_key = event_type.strip().lower()
        self._listeners.setdefault(event_key, []).append(callback)
        logger.debug("Subscribed callback to event '%s'", event_key)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unregister a callback for a specific event type."""
        event_key = event_type.strip().lower()
        if event_key in self._listeners:
            try:
                self._listeners[event_key].remove(callback)
                logger.debug("Unsubscribed callback from event '%s'", event_key)
            except ValueError:
                pass

    def publish(self, event_type: str, data: Any) -> None:
        """
        Synchronous publish bridge that schedules execution on the running loop
        or falls back to creating a task.
        """
        event_key = event_type.strip().lower()
        logger.debug("Publishing event '%s' (sync)", event_key)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish_async(event_type, data))
        except RuntimeError:
            # Fallback when no loop is running (e.g. startup/testing)
            asyncio.run(self.publish_async(event_type, data))

    async def publish_async(self, event_type: str, data: Any) -> None:
        """
        Asynchronously publish an event to all registered subscribers.
        Awaits any coroutine callbacks, and offloads blocking calls to threads.
        """
        event_key = event_type.strip().lower()
        listeners = list(self._listeners.get(event_key, []))
        # Support wildcard matching
        listeners.extend(self._listeners.get("*", []))

        tasks = []
        for listener in listeners:
            try:
                if inspect.iscoroutinefunction(listener):
                    tasks.append(listener(data))
                else:
                    tasks.append(asyncio.to_thread(listener, data))
            except Exception as e:
                logger.error("Error invoking subscriber for event '%s': %s", event_key, e, exc_info=True)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
