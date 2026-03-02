"""
core/hardware/device_registry.py
----------------------------------
Central registry for named hardware devices.
Devices are persisted to config/devices.json and instantiated lazily.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEVICES_PATH = Path("config/devices.json")


class DeviceRegistry:
    """Manages a collection of named SerialController instances."""

    def __init__(self):
        self._devices: dict = {}     # name -> config dict
        self._instances: dict = {}   # name -> SerialController
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load device configs from disk (silently ignore if missing/corrupt)."""
        if DEVICES_PATH.exists():
            try:
                self._devices = json.loads(DEVICES_PATH.read_text(encoding="utf-8"))
                logger.debug("Loaded %d device(s) from %s", len(self._devices), DEVICES_PATH)
            except Exception as e:
                logger.warning("devices.json load failed: %s", e)

    def _save(self) -> None:
        """Atomically write device configs to disk."""
        DEVICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEVICES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._devices, indent=2), encoding="utf-8")
        os.replace(tmp, DEVICES_PATH)
        logger.debug("Saved device registry to %s", DEVICES_PATH)

    # ── Device management ────────────────────────────────────────────────────

    def register_device(
        self,
        name: str,
        com_port: str,
        baud_rate: int = 115200,
        device_type: str = "arduino",
    ) -> None:
        """Register (or update) a named device.

        Args:
            name:        Logical device name, e.g. ``"main_arduino"``.
            com_port:    Serial port string, e.g. ``"COM7"`` or ``"SIM"`` for simulation.
            baud_rate:   Baud rate (default 115200).
            device_type: Free-form type label (e.g. ``"arduino"``, ``"sensor_hub"``).
        """
        self._devices[name] = {
            "com_port": com_port,
            "baud_rate": baud_rate,
            "device_type": device_type,
        }
        # Invalidate any cached instance so the next get_device() re-creates it.
        self._instances.pop(name, None)
        self._save()
        logger.info("Registered device '%s' on %s @ %d baud", name, com_port, baud_rate)

    def get_device(self, name: str):
        """Return (and lazily instantiate) the SerialController for *name*.

        Raises:
            KeyError: if the device has not been registered.
        """
        if name not in self._instances:
            if name not in self._devices:
                raise KeyError(f"Device '{name}' not registered")
            cfg = self._devices[name]
            from core.hardware.serial_controller import SerialController  # local import avoids cycles

            sim = cfg["com_port"].upper() == "SIM"
            # SerialController expects a config dict
            controller_cfg = {
                "com_port": cfg["com_port"],
                "baud_rate": cfg["baud_rate"],
                "timeout_seconds": cfg.get("timeout_seconds", 2),
                "require_hardware": "false",
            }
            ctrl = SerialController(controller_cfg)
            if sim:
                ctrl._simulation_mode = True  # force sim regardless of port check
            self._instances[name] = ctrl
            logger.info("Instantiated SerialController for '%s'", name)
        return self._instances[name]

    def list_devices(self) -> list:
        """Return a summary list of all registered devices."""
        result = []
        for name, cfg in self._devices.items():
            inst = self._instances.get(name)
            result.append(
                {
                    "name": name,
                    "port": cfg["com_port"],
                    "device_type": cfg.get("device_type", "unknown"),
                    "connected": inst is not None,
                    "simulation_mode": cfg["com_port"].upper() == "SIM",
                }
            )
        return result


__all__ = ["DeviceRegistry"]
