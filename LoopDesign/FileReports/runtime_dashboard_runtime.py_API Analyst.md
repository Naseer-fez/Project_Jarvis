# API Analyst Report: runtime\dashboard_runtime.py

## Dependencies
- `from __future__ import annotations`
- `import asyncio`
- `import contextlib`
- `import logging`
- `import threading`
- `import time`
- `from typing import Any`

## Schemas & API Contracts (Classes)

### Class `DashboardRuntime`
**Methods:**
- `def __init__(self, host: str, port: int, log: logging.Logger) -> None`
- `async def start(self, controller: Any, health_report: Any | None=None) -> None`
- `async def stop(self, timeout: float=5.0) -> None`

