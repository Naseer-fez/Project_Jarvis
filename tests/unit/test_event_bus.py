import asyncio
import threading
import time
import pytest
from core.runtime.event_bus import EventBus


def test_event_bus_basic_sync():
    """Verify synchronous subscription and publishing works correctly."""
    bus = EventBus(history_limit=10)
    received = []

    def on_event(data):
        received.append(data)

    bus.subscribe("test_event", on_event)
    bus.publish("test_event", "hello")
    bus.publish("test_event", "world")

    assert received == ["hello", "world"]
    assert len(bus.replay("test_event")) == 2
    assert bus.replay("test_event")[0].payload == "hello"


def test_event_bus_unsubscribe():
    """Verify that unsubscribing stops callbacks from firing."""
    bus = EventBus()
    received = []

    def on_event(data):
        received.append(data)

    bus.subscribe("test_event", on_event)
    bus.publish("test_event", "hello")
    bus.unsubscribe("test_event", on_event)
    bus.publish("test_event", "world")

    assert received == ["hello"]


@pytest.mark.asyncio
async def test_event_bus_async_callback_same_loop():
    """Verify async callbacks can be scheduled on the running event loop."""
    bus = EventBus()
    received = []
    future: asyncio.Future[bool] = asyncio.Future()

    async def on_event_async(data):
        received.append(data)
        if len(received) == 2:
            future.set_result(True)

    bus.subscribe("test_event", on_event_async)
    bus.publish("test_event", "async-1")
    bus.publish("test_event", "async-2")

    # Give loop a chance to process scheduled tasks
    await asyncio.wait_for(future, timeout=2.0)
    assert received == ["async-1", "async-2"]


@pytest.mark.asyncio
async def test_event_bus_publish_async():
    """Verify publish_async gathers and awaits all coroutines directly."""
    bus = EventBus()
    received = []

    async def on_event_async(data):
        await asyncio.sleep(0.01)
        received.append(data)

    bus.subscribe("test_event", on_event_async)
    record = await bus.publish_async("test_event", "async-data")

    assert record.payload == "async-data"
    assert received == ["async-data"]


def test_event_bus_reentrancy():
    """Verify that a callback can safely publish or subscribe within its execution (re-entrancy)."""
    bus = EventBus()
    received = []

    def inner_callback(data):
        received.append(f"inner:{data}")

    def outer_callback(data):
        received.append(f"outer:{data}")
        # Subscribe and publish inside callback - tests RLock re-entrancy
        bus.subscribe("inner_event", inner_callback)
        bus.publish("inner_event", f"reply-to-{data}")

    bus.subscribe("outer_event", outer_callback)
    bus.publish("outer_event", "hello")

    assert received == ["outer:hello", "inner:reply-to-hello"]


def test_event_bus_multithreaded_publish():
    """Verify thread-safety and lack of deadlocks under concurrent multithreaded publishing."""
    bus = EventBus(history_limit=2000)
    received = []
    lock = threading.Lock()

    def on_event(data):
        with lock:
            received.append(data)

    bus.subscribe("test_event", on_event)

    num_threads = 10
    publishes_per_thread = 100
    threads = []

    def worker(worker_id):
        for i in range(publishes_per_thread):
            bus.publish("test_event", f"w{worker_id}-{i}")

    for idx in range(num_threads):
        t = threading.Thread(target=worker, args=(idx,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(received) == num_threads * publishes_per_thread
    assert len(bus.replay("test_event")) == num_threads * publishes_per_thread


@pytest.mark.asyncio
async def test_event_bus_cross_thread_async_dispatch():
    """Verify scheduling async callbacks thread-safely from a background thread to main loop."""
    bus = EventBus()
    received = []
    event = asyncio.Event()

    async def on_event_async(data):
        received.append(data)
        event.set()

    bus.subscribe("test_event", on_event_async)

    # Run publisher in background thread
    def background_publisher():
        time.sleep(0.1)
        bus.publish("test_event", "cross-thread-data")

    t = threading.Thread(target=background_publisher)
    t.start()

    await asyncio.wait_for(event.wait(), timeout=2.0)
    t.join()

    assert received == ["cross-thread-data"]


def test_event_bus_callback_exception_handling():
    """Verify callback exception handling so that one failing subscriber doesn't halt others."""
    bus = EventBus()
    received = []

    def bad_callback(data):
        raise ValueError("Simulated failure")

    def good_callback(data):
        received.append(data)

    bus.subscribe("test_event", bad_callback)
    bus.subscribe("test_event", good_callback)

    # Should not raise exception
    bus.publish("test_event", "data")
    assert received == ["data"]
