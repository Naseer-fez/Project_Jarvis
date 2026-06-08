"""
core/hardware/device_registry.py
--------------------------------
Registry of hardware devices, supporting dynamic loading and integration with SerialController.
"""

from __future__ import annotations
import logging
import asyncio
from typing import Dict, List, Any
from core.hardware.serial_controller import SerialController

logger = logging.getLogger(__name__)

class HardwareDevice:
    def __init__(self, name: str, controller: SerialController) -> None:
        self.name = name
        self.controller = controller

    async def async_send_command(self, command: str, value: str = "") -> str:
        """Send command to device asynchronously using a thread pool."""
        if not self.controller.enabled:
            raise NotImplementedError(f"Hardware serial control is disabled for {self.name}.")
        
        full_command = f"{command} {value}".strip()
        # Offload the blocking serial send to a thread pool
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.controller.send, full_command)
            return str(result)
        except Exception as e:
            logger.error("Failed to send command to %s: %s", self.name, e, exc_info=True)
            raise

    async def firmware_ping(self) -> bool:
        """Ping the device firmware."""
        try:
            res = await self.async_send_command("PING")
            return "PONG" in res or "OK" in res or res != ""
        except Exception:
            return False


class DeviceRegistry:
    def __init__(self) -> None:
        self._devices: Dict[str, HardwareDevice] = {}
        self._load_from_config()

    def _load_from_config(self) -> None:
        try:
            controller = SerialController()
            # If enabled, register it
            if controller.enabled:
                self.register_device("main_arduino", controller)
        except Exception as e:
            logger.error("Failed to load hardware devices from config: %s", e, exc_info=True)

    def register_device(self, name: str, controller: SerialController) -> None:
        self._devices[name] = HardwareDevice(name, controller)

    def get_device(self, name: str) -> HardwareDevice:
        if name not in self._devices:
            logger.warning("Device '%s' not registered. Creating a default disabled device.", name)
            controller = SerialController()  # Disabled by default
            self._devices[name] = HardwareDevice(name, controller)
        return self._devices[name]

    def list_devices(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "connected": device.controller.is_connected,
                "enabled": device.controller.enabled,
            }
            for name, device in self._devices.items()
        ]
