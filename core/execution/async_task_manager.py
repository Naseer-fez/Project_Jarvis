"""
Priority-aware async task manager with cancellation support.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass
class TaskRecord:
    task_id: str
    name: str
    priority: int
    status: str
    error: str | None = None


class AsyncTaskManager:
    def __init__(self, max_parallel: int = 3) -> None:
        self._max_parallel = max(1, int(max_parallel))
        self._pending: list[tuple[int, int, str, str, Callable[[], Awaitable[object]], asyncio.Future]] = []
        self._running: dict[str, asyncio.Task] = {}
        self._records: dict[str, TaskRecord] = {}
        self._counter = itertools.count()
        self._semaphore = asyncio.Semaphore(self._max_parallel)
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop(), name="task_manager_scheduler")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        for task_id, task in list(self._running.items()):
            task.cancel()
            self._records[task_id].status = "cancelled"
        self._running.clear()

    async def submit(
        self,
        coro_factory: Callable[[], Awaitable[object]],
        *,
        priority: int = 5,
        name: str = "task",
    ) -> tuple[str, asyncio.Future]:
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        heapq.heappush(
            self._pending,
            (int(priority), next(self._counter), task_id, name, coro_factory, future),
        )
        self._records[task_id] = TaskRecord(
            task_id=task_id,
            name=name,
            priority=int(priority),
            status="pending",
        )
        return task_id, future

    def cancel(self, task_id: str) -> bool:
        if task_id in self._running:
            self._running[task_id].cancel()
            rec = self._records.get(task_id)
            if rec:
                rec.status = "cancelled"
            return True

        for idx, pending in enumerate(list(self._pending)):
            _, _, pid, _, _, fut = pending
            if pid != task_id:
                continue
            self._pending.pop(idx)
            heapq.heapify(self._pending)
            if not fut.done():
                fut.cancel()
            rec = self._records.get(task_id)
            if rec:
                rec.status = "cancelled"
            return True
        return False

    def snapshot(self) -> list[TaskRecord]:
        return [self._records[k] for k in sorted(self._records.keys())]

    async def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(0.01)
            if not self._pending:
                continue
            if self._semaphore.locked() and len(self._running) >= self._max_parallel:
                continue

            priority, _, task_id, name, factory, future = heapq.heappop(self._pending)
            rec = self._records[task_id]
            rec.status = "running"

            task = asyncio.create_task(
                self._run_one(task_id, name, factory, future),
                name=f"{name}:{task_id}:{priority}",
            )
            self._running[task_id] = task

    async def _run_one(
        self,
        task_id: str,
        _name: str,
        factory: Callable[[], Awaitable[object]],
        future: asyncio.Future,
    ) -> None:
        async with self._semaphore:
            try:
                result = await factory()
                if not future.done():
                    future.set_result(result)
                self._records[task_id].status = "completed"
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                self._records[task_id].status = "cancelled"
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                self._records[task_id].status = "failed"
                self._records[task_id].error = str(exc)
            finally:
                self._running.pop(task_id, None)
