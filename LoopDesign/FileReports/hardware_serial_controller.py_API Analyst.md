# API Analyst Report: hardware\serial_controller.py

## Dependencies
- `from __future__ import annotations`

## Schemas & API Contracts (Classes)

### Class `SerialController`
**Methods:**
- `def __init__(self, config=None, port: str | None=None, baud_rate: int | None=None, timeout: float=1.0) -> None`
- @property
- `def is_connected(self) -> bool`
- `def connect(self, port: str | None=None)`
- `def send(self, command: str)`
- `def close(self) -> None`

