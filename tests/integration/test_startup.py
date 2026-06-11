import asyncio
import configparser
from types import SimpleNamespace

import pytest

from core.controller_v2 import JarvisControllerV2
from core.runtime.bootstrap import _cancel_task, _validate_startup_settings
from core.runtime.paths import PROJECT_ROOT, _resolve_path

@pytest.mark.asyncio
async def test_controller_startup_and_shutdown(tmp_path):
    """Verify that JarvisControllerV2 can start up and shut down without errors."""
    config = configparser.ConfigParser()
    config.add_section("memory")
    db_file = tmp_path / "test_memory.db"
    config.set("memory", "db_path", str(db_file))
    
    controller = JarvisControllerV2(config=config, voice=False)
    
    # Verify components are initialized
    assert controller.memory is not None
    assert controller.model_router is not None
    assert controller.profile is not None
    assert controller.llm is not None
    
    # Start the controller (initializes memory, starts background monitor, etc)
    await controller.start()
    
    # Ensure memory initialized correctly
    assert controller.memory.mode in ["hybrid", "chroma", "sqlite", "sqlite-only", "null"]
    
    # Shutdown the controller
    await controller.shutdown()


def test_resolve_path_normalizes_relative_segments():
    resolved = _resolve_path("logs/../outputs")

    assert resolved == (PROJECT_ROOT / "outputs").resolve(strict=False)


def test_validate_startup_settings_rejects_invalid_dashboard_port():
    config = configparser.ConfigParser()
    config.add_section("dashboard")
    config.set("dashboard", "host", "127.0.0.1")
    config.set("dashboard", "port", "not-a-port")

    args = SimpleNamespace(
        verify=False,
        health_check=False,
        dashboard_host=None,
        dashboard_port=None,
    )

    validation = _validate_startup_settings(
        config,
        args,  # type: ignore[arg-type]
        voice_enabled=False,
        dashboard_enabled=True,
        headless=False,
        shutdown_timeout=5.0,
    )

    assert not validation.is_valid
    assert "Dashboard port must be between 1 and 65535." in validation.errors


@pytest.mark.asyncio
async def test_cancel_task_handles_completed_cancelled_task():
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await _cancel_task(task)
