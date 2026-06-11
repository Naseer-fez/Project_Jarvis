import asyncio
import pytest
import configparser
from core.controller_v2 import JarvisControllerV2
from core.memory.hybrid_memory import HybridMemory
from core.memory.sqlite_pool import SQLitePool

@pytest.mark.asyncio
async def test_resource_cleanup_during_shutdown(tmp_path):
    """Verify that database connections and background tasks are gracefully cleaned up on shutdown."""
    config = configparser.ConfigParser()
    config.add_section("memory")
    db_file = tmp_path / "test_cleanup_memory.db"
    config.set("memory", "db_path", str(db_file))
    
    # Enable a simulated/temp automation configuration if needed, but voice=False is standard
    controller = JarvisControllerV2(config=config, voice=False)
    
    # Initialize and start components
    await controller.start()
    
    # Check that sqlite pool exists and is active
    pool = controller.memory._pool
    assert pool is not None
    assert not pool._closed
    
    # Trigger a dummy read/write to acquire connection and verify the pool initializes
    await controller.memory.store_preference("test_key", "test_val")
    assert pool._pool is not None  # Queue of connections initialized
    assert len(pool._all_conns) > 0  # Connections tracked
    
    # Verify goal checker task is running
    assert controller._goal_check_task is not None
    assert not controller._goal_check_task.done()
    
    # Verify monitor is running
    assert controller.monitor._running
    
    # Shut down the controller
    await controller.shutdown()
    
    # Verify SQLite pool is closed and connections are cleared
    assert pool._closed
    assert pool._pool is None
    assert len(pool._all_conns) == 0
    
    # Verify goal check task is done/cancelled
    assert controller._goal_check_task.done()
    
    # Verify monitor task is stopped
    assert not controller.monitor._running
    assert len(controller.monitor._tasks) == 0


@pytest.mark.asyncio
async def test_hybrid_memory_close_cancels_background_index_task(tmp_path):
    """Verify background indexing tasks are cancelled before the pool is closed."""
    memory = HybridMemory(db_path=str(tmp_path / "hybrid_cleanup.db"))
    started = asyncio.Event()
    released = asyncio.Event()

    async def slow_index(_root_path: str) -> dict[str, int]:
        started.set()
        await released.wait()
        return {"indexed_files": 0, "indexed_chunks": 0, "skipped_files": 0, "errors": 0}

    memory.index_codebase = slow_index  # type: ignore[method-assign,assignment]

    result = await memory.initialize(index_path=str(tmp_path))
    assert result["codebase_index"]["status"] == "background_indexing_started"

    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert len(memory._background_tasks) == 1
    task = next(iter(memory._background_tasks))

    await memory.close()

    assert task.cancelled()
    assert len(memory._background_tasks) == 0
    assert memory._pool._closed
    assert memory._pool._pool is None

    released.set()


@pytest.mark.asyncio
async def test_sqlite_pool_close_waits_for_checked_out_connection(tmp_path):
    """Verify pool close waits for borrowed connections to cleanly return."""
    pool = SQLitePool(str(tmp_path / "pool_cleanup.db"), pool_size=1)
    acquired = asyncio.Event()
    release_conn = asyncio.Event()

    async def hold_connection() -> None:
        async with pool.acquire() as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY)")
            acquired.set()
            await release_conn.wait()

    holder = asyncio.create_task(hold_connection())
    await asyncio.wait_for(acquired.wait(), timeout=1.0)

    close_task = asyncio.create_task(pool.close())
    await asyncio.sleep(0.05)
    assert not close_task.done()

    release_conn.set()
    await close_task
    await holder

    assert pool._closed
    assert pool._pool is None
    assert len(pool._all_conns) == 0
    assert len(pool._in_use_conns) == 0
