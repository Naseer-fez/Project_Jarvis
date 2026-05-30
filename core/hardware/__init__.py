"""Hardware compatibility package."""

from .serial_controller import SerialController
from .device_registry import DeviceRegistry, HardwareDevice

__all__ = ["SerialController", "DeviceRegistry", "HardwareDevice"]
