"""
core/tools/hardware_tools.py
------------------------------
Async tool functions for interacting with registered hardware devices via
the DeviceRegistry / SerialController layer.

All functions return a ToolResult so they integrate cleanly with the ToolRouter.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy singleton — created once on first import.
_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        from core.hardware.device_registry import DeviceRegistry
        _registry = DeviceRegistry()
    return _registry


# ── Tool functions ────────────────────────────────────────────────────────────

async def send_hardware_command(
    device_name: str, command: str, value: str = ""
):
    """Send an arbitrary command to a registered hardware device.

    Args:
        device_name: Registered device name (e.g. ``"main_arduino"``).
        command:     Command string (e.g. ``"LIGHT"``).
        value:       Optional value string (e.g. ``"ON"``).
    """
    from integrations.base import ToolResult
    try:
        device = _get_registry().get_device(device_name)
        result = await device.async_send_command(command, value)
        return ToolResult(success=True, data=result)
    except Exception as e:
        logger.error("send_hardware_command failed: %s", e)
        return ToolResult(success=False, error=str(e))


async def read_sensor(device_name: str, sensor_type: str = "all"):
    """Request a sensor reading from a registered device.

    Args:
        device_name: Registered device name.
        sensor_type: Sensor identifier (e.g. ``"TEMPERATURE"``). Defaults to ``"all"``.
    """
    from integrations.base import ToolResult
    try:
        device = _get_registry().get_device(device_name)
        result = await device.async_send_command("READ", sensor_type)
        return ToolResult(success=True, data=result)
    except Exception as e:
        logger.error("read_sensor failed: %s", e)
        return ToolResult(success=False, error=str(e))


async def list_hardware_devices():
    """Return a list of all registered hardware devices with their status."""
    from integrations.base import ToolResult
    try:
        devices = _get_registry().list_devices()
        return ToolResult(success=True, data={"devices": devices})
    except Exception as e:
        logger.error("list_hardware_devices failed: %s", e)
        return ToolResult(success=False, error=str(e))


async def ping_device(device_name: str):
    """Ping a registered device to check if its firmware is responsive.

    Args:
        device_name: Registered device name.
    """
    from integrations.base import ToolResult
    try:
        device = _get_registry().get_device(device_name)
        alive = await device.firmware_ping()
        return ToolResult(success=True, data={"alive": alive, "device": device_name})
    except Exception as e:
        logger.error("ping_device failed: %s", e)
        return ToolResult(success=False, error=str(e))


__all__ = [
    "send_hardware_command",
    "read_sensor",
    "list_hardware_devices",
    "ping_device",
]
