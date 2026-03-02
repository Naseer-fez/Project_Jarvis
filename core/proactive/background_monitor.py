import asyncio
import logging

logger = logging.getLogger(__name__)


class BackgroundMonitor:
    def __init__(self, notifier, config=None):
        self.notifier = notifier
        self.cpu_threshold = 90
        self.ram_threshold = 90
        if config and config.has_section("proactive"):
            self.cpu_threshold = config.getint("proactive", "cpu_alert_threshold", fallback=90)
            self.ram_threshold = config.getint("proactive", "ram_alert_threshold", fallback=90)
        self._tasks: list = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._monitor_resources()))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    async def _monitor_resources(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            try:
                import psutil

                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory().percent
                if cpu > self.cpu_threshold:
                    self.notifier.notify(f"\u26a0\ufe0f CPU at {cpu:.0f}%", level="warn")
                if ram > self.ram_threshold:
                    self.notifier.notify(f"\u26a0\ufe0f RAM at {ram:.0f}%", level="warn")
            except ImportError:
                pass
