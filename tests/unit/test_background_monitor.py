import pytest
from core.proactive.background_monitor import BackgroundMonitor


class MockNotifier:
    def __init__(self):
        self.notifications = []

    def notify(self, message, level="info"):
        self.notifications.append((message, level))


@pytest.mark.asyncio
async def test_background_monitor_lifecycle():
    """Verify that BackgroundMonitor initializes, starts, and stops cleanly."""
    notifier = MockNotifier()
    monitor = BackgroundMonitor(notifier)

    assert monitor.cpu_threshold == 90
    assert monitor.ram_threshold == 90
    assert not monitor._running
    assert len(monitor._tasks) == 0

    # Start the monitor
    await monitor.start()
    assert monitor._running
    assert len(monitor._tasks) == 1

    # Stop the monitor and verify tasks are cleaned up
    await monitor.stop()
    assert not monitor._running
    assert len(monitor._tasks) == 0


@pytest.mark.asyncio
async def test_background_monitor_cancellation_await():
    """Verify that BackgroundMonitor.stop() waits for its tasks to finish cancelling."""
    notifier = MockNotifier()
    monitor = BackgroundMonitor(notifier)

    await monitor.start()
    task = monitor._tasks[0]
    assert not task.done()

    await monitor.stop()
    assert task.done()
    assert task.cancelled()
