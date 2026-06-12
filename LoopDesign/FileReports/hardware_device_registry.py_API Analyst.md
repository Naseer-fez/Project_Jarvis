# API Analyst Report: hardware\device_registry.py

## Dependencies
- `from __future__ import annotations`
- `import logging`
- `import asyncio`
- `from typing import Dict`
- `from typing import List`
- `from typing import Any`
- `from core.hardware.serial_controller import SerialController`

## Schemas & API Contracts (Classes)

### Class `HardwareDevice`
**Methods:**
- `def __init__(self, name: str, controller: SerialController) -> None`
- `async def async_send_command(self, command: str, value: str='') -> str`
  - *Send command to device asynchronously using a thread pool.*
- `async def firmware_ping(self) -> bool`
  - *Ping the device firmware.*


### Class `DeviceRegistry`
**Methods:**
- `def __init__(self) -> None`
- `def _load_from_config(self) -> None`
- `def register_device(self, name: str, controller: SerialController) -> None`
- `def get_device(self, name: str) -> HardwareDevice`
- `def list_devices(self) -> List[Dict[str, Any]]`

