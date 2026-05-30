"""Lightweight pub/sub event bus for decoupled component communications."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger("Jarvis.EventBus")

EventCallback = Union[Callable[[Any], None], Callable[[Any], Awaitable[None]]]


@dataclass(frozen=True)
class EventRecord:
    """Replayable event envelope stored by the local event bus."""

    event_id: str
    event_type: str
    payload: Any
    source: str = "runtime"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "created_at": self.created_at,
        }


class EventBus:
    """
    Publish/Subscribe Event Bus allowing loose coupling between modules.
    """

    def __init__(self, *, history_limit: int = 500) -> None:
        self._listeners: dict[str, list[EventCallback]] = {}
        self._history: deque[EventRecord] = deque(maxlen=max(0, int(history_limit)))

    def subscribe(
        self,
        event_type: str,
        callback: EventCallback,
        *,
        replay_history: bool = False,
    ) -> None:
        """Register a callback for a specific event type."""
        event_key = event_type.strip().lower()
        if event_key not in self._listeners:
            self._listeners[event_key] = []
        if callback not in self._listeners[event_key]:
            self._listeners[event_key].append(callback)
            logger.debug("Subscribed callback to event '%s'", event_key)
        if replay_history:
            for record in self.replay(event_key):
                self._dispatch_callback(callback, record.payload, event_key)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Unregister a callback for a specific event type."""
        event_key = event_type.strip().lower()
        if event_key in self._listeners:
            try:
                self._listeners[event_key].remove(callback)
                logger.debug("Unsubscribed callback from event '%s'", event_key)
            except ValueError:
                pass

    def publish(self, event_type: str, data: Any, *, source: str = "runtime") -> EventRecord:
        """
        Publish an event to all registered subscribers.
        Dispatches both synchronous and asynchronous callbacks safely.
        """
        record = self._record(event_type, data, source=source)

        for callback in self._callbacks_for(record.event_type):
            self._dispatch_callback(callback, data, record.event_type)
        return record

    async def publish_async(self, event_type: str, data: Any, *, source: str = "runtime") -> EventRecord:
        """
        Asynchronously publish an event to all registered subscribers.
        Awaits any coroutine callbacks.
        """
        record = self._record(event_type, data, source=source)

        tasks = []
        for callback in self._callbacks_for(record.event_type):
            try:
                res = callback(data)
                if asyncio.iscoroutine(res):
                    tasks.append(res)
            except Exception as e:
                logger.error("Error in async subscriber callback for event '%s': %s", record.event_type, e, exc_info=True)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return record

    def replay(self, event_type: str | None = None, *, limit: int | None = None) -> list[EventRecord]:
        """Return recent events, optionally filtered by type."""
        event_key = event_type.strip().lower() if event_type else None
        items = [
            record
            for record in self._history
            if event_key in (None, "*") or record.event_type == event_key
        ]
        if limit is not None:
            return items[-max(0, int(limit)) :]
        return items

    def clear_history(self) -> None:
        self._history.clear()

    def _record(self, event_type: str, payload: Any, *, source: str) -> EventRecord:
        event_key = event_type.strip().lower()
        record = EventRecord(
            event_id=uuid.uuid4().hex,
            event_type=event_key,
            payload=payload,
            source=source,
        )
        if self._history.maxlen != 0:
            self._history.append(record)
        return record

    def _callbacks_for(self, event_key: str) -> list[EventCallback]:
        callbacks: list[EventCallback] = []
        callbacks.extend(self._listeners.get(event_key, []))
        callbacks.extend(self._listeners.get("*", []))
        return list(callbacks)

    def _dispatch_callback(self, callback: EventCallback, data: Any, event_key: str) -> None:
        try:
            res = callback(data)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(res)
                except RuntimeError:
                    logger.warning(
                        "Async subscriber callback for event '%s' called without a running event loop. "
                        "Running coroutine synchronously using asyncio.run() fallback.",
                        event_key
                    )
                    asyncio.run(res)
        except Exception as e:
            logger.error("Error in subscriber callback for event '%s': %s", event_key, e, exc_info=True)


__all__ = ["EventBus", "EventCallback", "EventRecord"]
