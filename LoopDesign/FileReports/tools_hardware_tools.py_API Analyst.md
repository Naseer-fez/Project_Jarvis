# API Analyst Report: tools\hardware_tools.py

## Dependencies
- `from __future__ import annotations`
- `from core.types.common import ToolResult`
- `import logging`

## Functions & Endpoints

### `_get_registry`
`def _get_registry()`
### `send_hardware_command`
`async def send_hardware_command(device_name: str, command: str, value: str='')`
> Send an arbitrary command to a registered hardware device.

Args:
    device_name: Registered device name (e.g. ``"main_arduino"``).
    command:     Command string (e.g. ``"LIGHT"``).
    value:       Optional value string (e.g. ``"ON"``).

### `read_sensor`
`async def read_sensor(device_name: str, sensor_type: str='all')`
> Request a sensor reading from a registered device.

Args:
    device_name: Registered device name.
    sensor_type: Sensor identifier (e.g. ``"TEMPERATURE"``). Defaults to ``"all"``.

### `list_hardware_devices`
`async def list_hardware_devices()`
> Return a list of all registered hardware devices with their status.

### `ping_device`
`async def ping_device(device_name: str)`
> Ping a registered device to check if its firmware is responsive.

Args:
    device_name: Registered device name.
